#!/bin/bash

echo "=== Instalando dependências do Transcritor ==="

# Instalar ffmpeg
echo "Instalando ffmpeg..."
sudo apt update && sudo apt install ffmpeg -y

# Criar ambiente virtual
echo "Criando ambiente virtual..."
cd ~/projects/transcriptor
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate

# Instalar dependências Python
echo "Instalando Python packages..."
pip install -r requirements.txt

echo ""
echo "=== Instalação concluída! ==="
echo ""
echo "Para executar a aplicação:"
echo "  cd ~/projects/transcriptor"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "Acesse: http://localhost:8501"
