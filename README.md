# codator — Dev-Assistant-OS

Hybrydowy asystent kodowania AI uruchamiany lokalnie na Ubuntu Server 24.04 z AMD Radeon RX 9070 XT (ROCm) + opcjonalne API chmurowe (Claude/OpenAI).

## Cechy

- **Lokalne modele LLM** via `llama-cpp-python` z ROCm/HIP — pełne wykorzystanie 16 GB VRAM
- **Inteligentne zarządzanie kontekstem** — `AdaptiveContextManager` z automatycznym kompaktowaniem (Summary Snapshot)
- **Zero-Hallucination Policy** — model prosi o brakujące pliki zamiast zgadywać
- **Indeksowanie projektu** — tree-sitter mapa symboli (funkcje, klasy, metody)
- **Integracja z Git** — `git diff` jako priorytetowy kontekst
- **CLI** z `rich` (syntax highlighting) + `prompt_toolkit` (autouzupełnianie)
- **Web Dashboard** (FastAPI) — podgląd VRAM/temp GPU, zmiana modeli w locie

## Wymagania systemowe

| Komponent | Minimum |
|-----------|---------|
| OS | Ubuntu 24.04 LTS |
| GPU | AMD Radeon RX 9070 XT (RDNA 4, 16 GB VRAM) |
| RAM | 32 GB DDR5 |
| CPU | AMD Ryzen 9 |
| ROCm | 6.x |
| Python | 3.11+ |

## Instalacja

```bash
git clone <repo-url> && cd codator
chmod +x install.sh
./install.sh
```

Skrypt automatycznie:
1. Instaluje ROCm 6.x (jeśli brak)
2. Tworzy venv Python 3.12
3. Kompiluje `llama-cpp-python` z backendem HIP (target `gfx1201`)
4. Instaluje codator z zależnościami

### Ręczna instalacja (krok po kroku)

```bash
# 1. ROCm (patrz: https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html)
sudo apt install rocm-hip-sdk rocm-smi-lib

# 2. Python venv
python3.12 -m venv .venv && source .venv/bin/activate

# 3. llama-cpp-python z ROCm
CMAKE_ARGS="-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201" pip install llama-cpp-python

# 4. codator
pip install -e ".[dev]"
```

## Użycie

### CLI

```bash
source .venv/bin/activate

# Z lokalnym modelem:
codator --model /path/to/deepseek-coder-7b-Q8_0.gguf

# Z API chmurowym:
export ANTHROPIC_API_KEY="sk-ant-..."
codator --api claude

# Z dashboardem web:
codator --model /path/to/model.gguf --web
```

### Komendy w czacie

| Komenda | Opis |
|---------|------|
| `/help` | Pomoc |
| `/model <path>` | Załaduj model GGUF |
| `/api <claude\|openai>` | Przełącz na API |
| `/context` | Status kontekstu (tokeny, kompaktowania) |
| `/clear` | Wyczyść rozmowę |
| `/hardware` | Info o GPU/RAM |
| `/index` | Przeindeksuj projekt |
| `/git` | Pokaż kontekst git |
| `/web` | Uruchom dashboard web |

### Web Dashboard

Po uruchomieniu z `--web` lub komendą `/web`, dashboard dostępny na `http://localhost:8000`:
- Podgląd aktywnego modelu i zmiana w locie
- Pasek użycia kontekstu (tokeny)
- Status GPU: VRAM, temperatura, wersja ROCm
- Rekomendacje modeli dla Twojego hardware

## Architektura

```
src/codator/
├── domain/              # Modele danych, interfejsy (0 zależności)
│   ├── models.py        # Message, HardwareInfo, ProjectMap, ...
│   └── interfaces.py    # ABC: InferenceBackend, ContextManager, ...
├── infrastructure/      # Implementacje niskopoziomowe
│   ├── hardware.py      # Detekcja GPU/ROCm via rocm-smi
│   ├── inference.py     # llama-cpp-python backend
│   ├── api_clients.py   # Claude + OpenAI klienty
│   └── tokenizer.py     # Licznik tokenów (tiktoken / llama)
├── core/                # Logika biznesowa
│   ├── context_manager.py  # AdaptiveContextManager + kompaktowanie
│   ├── project_indexer.py  # tree-sitter skaner symboli
│   ├── git_integration.py  # git diff / status
│   └── chat_engine.py      # Orkiestrator łączący wszystko
├── cli/                 # Warstwa prezentacji CLI
│   ├── app.py           # Główna pętla async + prompt_toolkit
│   ├── commands.py      # Slash-command handler
│   └── rendering.py     # Rich rendering (panele, tabele, MD)
└── web/                 # Warstwa prezentacji Web
    ├── app.py           # FastAPI + endpoints
    └── static/index.html # Dashboard SPA
```

## Kompaktowanie kontekstu

Gdy użycie tokenów przekroczy 80% okna kontekstowego:

1. **Generacja Summary Snapshot** — mały model 3B streszcza dotychczasową rozmowę
2. **Zachowanie** — system prompt + snapshot + 2 ostatnie wiadomości
3. **Powiadomienie** — użytkownik widzi info o kompaktowaniu w CLI
4. **Przejrzystość** — `/context` pokazuje historię kompaktowań

Kontekst **nigdy** nie jest cicho ucinany.

## Licencja

MIT
