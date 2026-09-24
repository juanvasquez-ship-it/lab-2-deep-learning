"""FDMLP de las ecuaciones 11–15, LSTM base y ablación de CReLU."""

import torch
from torch import nn


class CapaCompleja(nn.Module):
    """Producto complejo elemento a elemento, según la ecuación 11."""

    def __init__(self, bins):
        super().__init__()
        self.peso_real = nn.Parameter(torch.ones(bins))
        self.peso_imag = nn.Parameter(torch.zeros(bins))
        self.sesgo_real = nn.Parameter(torch.zeros(bins))
        self.sesgo_imag = nn.Parameter(torch.zeros(bins))

    def forward(self, real, imag):
        return (real * self.peso_real - imag * self.peso_imag + self.sesgo_real,
                real * self.peso_imag + imag * self.peso_real + self.sesgo_imag)


class FDMLP(nn.Module):
    def __init__(self, variables, nonlinear=True):
        super().__init__()
        self.variables = variables
        self.nonlinear = nonlinear
        self.entrada = CapaCompleja(variables // 2 + 1)
        self.salida = CapaCompleja(variables // 2 + 1)
        self.capturar_espectro = False

    def forward(self, x):
        # La FFT opera entre variables de un mismo instante, nunca entre horas.
        # float32 evita las restricciones de cuFFT en media precisión para N=12.
        with torch.autocast(device_type=x.device.type, enabled=False):
            espectro = torch.fft.rfft(x.float(), dim=-1, norm="ortho")
            if self.capturar_espectro:
                espectro.retain_grad()
                self.ultimo_espectro = espectro
            real, imag = self.entrada(espectro.real, espectro.imag)
            if self.nonlinear:
                real, imag = torch.relu(real), torch.relu(imag)
            real, imag = self.salida(real, imag)
            # irfft reconstruye una señal real con simetría conjugada implícita.
            return torch.fft.irfft(torch.complex(real, imag), n=self.variables,
                                   dim=-1, norm="ortho")


class AtencionVariables(nn.Module):
    """Ponderación por variable con contexto recurrente, ecuaciones 2–5."""

    def __init__(self, variables, context_hidden=32):
        super().__init__()
        self.contexto = nn.LSTM(variables, context_hidden, batch_first=True, bidirectional=True)
        self.peso_contexto = nn.Linear(2*context_hidden, variables, bias=False)
        self.peso_entrada = nn.Linear(variables, variables, bias=False)

    def pesos(self, x):
        contexto, _ = self.contexto(x)
        # El contexto bidireccional solo ve las 336 horas históricas disponibles.
        anterior_h = torch.cat((torch.zeros_like(contexto[:, :1]), contexto[:, :-1]), dim=1)
        anterior_x = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
        return torch.softmax(torch.tanh(self.peso_contexto(anterior_h) + self.peso_entrada(anterior_x)), dim=-1)

    def forward(self, x):
        return x * self.pesos(x)


class ConvolucionVariables(nn.Module):
    """Convoluciones sobre variables por instante y pooling global, ecuaciones 6–7."""

    def __init__(self, variables, channels=32, kernel_size=3):
        super().__init__()
        self.capas = nn.Sequential(nn.Conv1d(1, channels, kernel_size, padding=kernel_size//2),
                                   nn.ReLU(),
                                   nn.Conv1d(channels, variables, kernel_size, padding=kernel_size//2),
                                   nn.ReLU())

    def forward(self, x):
        b, t, n = x.shape
        features = self.capas(x.reshape(b*t, 1, n)).mean(dim=-1)
        return features.reshape(b, t, n)


class GrafoVariables(nn.Module):
    """Grafo aprendido entre canales, simétrico y no negativo, ecuación 8."""

    def __init__(self, variables, node_hidden=16):
        super().__init__()
        self.adyacencia_libre = nn.Parameter(torch.empty(variables, variables).normal_(-2, .1))
        self.entrada = nn.Linear(1, node_hidden, bias=False)
        self.salida = nn.Linear(node_hidden, 1, bias=False)
        # Evita iniciar con todos los nodos apagados por la segunda ReLU.
        nn.init.constant_(self.salida.weight, 1/node_hidden)

    def adyacencia(self):
        a = torch.nn.functional.softplus(self.adyacencia_libre)
        a = (a+a.T)/2
        a = a + torch.eye(len(a), device=a.device, dtype=a.dtype)
        inv_degree = a.sum(dim=1).rsqrt()
        return inv_degree[:, None]*a*inv_degree[None, :]

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            a = self.adyacencia()
            h = torch.relu(self.entrada(torch.einsum("ij,btj->bti", a, x.float()).unsqueeze(-1)))
            h = torch.einsum("ij,btjd->btid", a, h)
            return torch.relu(self.salida(h)).squeeze(-1)


class Pronosticador(nn.Module):
    def __init__(self, kind, variables, hidden_size=128, layers=2, horizon=48, feature_config=None):
        super().__init__()
        # Se inicializa primero el mismo tronco para igualar el punto de partida.
        self.lstm = nn.LSTM(variables, hidden_size, num_layers=layers, batch_first=True)
        self.salida = nn.Linear(hidden_size, horizon)
        feature_config = feature_config or {}
        if kind in ("lstm", "lstm_uni"):
            self.caracteristicas = nn.Identity()
        elif kind in ("fdmlp", "fdmlp_lineal"):
            self.caracteristicas = FDMLP(variables, nonlinear=kind == "fdmlp")
        elif kind == "am":
            self.caracteristicas = AtencionVariables(variables, **feature_config)
        elif kind == "cnn":
            self.caracteristicas = ConvolucionVariables(variables, **feature_config)
        elif kind == "gnn":
            self.caracteristicas = GrafoVariables(variables, **feature_config)
        else:
            raise ValueError(f"Modelo desconocido: {kind}")

    def forward(self, x):
        _, (hidden, _) = self.lstm(self.caracteristicas(x))
        return self.salida(hidden[-1])
