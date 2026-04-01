# Transcritor Pro - Whisper + Embeddings + Chat

## Recursos

- **Upload de múltiplos arquivos** (mp4, mp3, wav, m4a, ogg, avi, mkv)
- **Suporte a arquivos grandes** (500MB+)
- **Verificador de modelo** - verifica se modelo já está em cache
- **Fila de execução** - status em tempo real
- **Validação de transcrição** - edite antes de vetorizar
- **Embeddings** - vetorização com sentence-transformers
- **Chat de estudos** - pergunte sobre o conteúdo transcrito

## Instalação

```bash
sudo apt install ffmpeg python3.12-venv python3-pip
cd ~/projects/transcriptor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Executar

```bash
./start_transcriptor
```

Acesse: http://localhost:8501

## Abas

1. **📤 Upload & Fila** - Envie arquivos, acompanhe processamento
2. **📝 Validação** - Revise e edite transcrições
3. **💬 Chat de Estudos** - Faça perguntas sobre o conteúdo vetorizado

## Estrutura

```
transcriptor/
├── app.py              # App Streamlit
├── lib/
│   ├── model_manager.py   # Gerenciamento de modelos Whisper
│   ├── queue_manager.py   # Fila de processamento
│   └── embeddings.py      # Geração de embeddings FAISS
├── transcripts/        # Transcrições salvas
│   └── embeddings/    # Índice FAISS
└── .devorq/           # Framework DEVORQ
```

## Modelos Whisper

| Modelo | VRAM | Velocidade |
|--------|------|------------|
| tiny   | ~1GB | 10x        |
| base   | ~1GB | 7x         |
| small  | ~2GB | 4x         |
| medium | ~5GB | 2x         |
| large  | ~10GB| 1x         |
| turbo  | ~6GB | 8x         |

## Embeddings

- **Modelo**: all-MiniLM-L6-v2 (leve, ~90MB)
- **Índice**: FAISS (busca rápida por similaridade)
