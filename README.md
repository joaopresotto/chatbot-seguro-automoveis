# Chatbot Seguro Automóveis

## Descrição
Este projeto é um chatbot desenvolvido para responder dúvidas sobre seguros de automóveis das seguradoras Santander, Bradesco, Porto Seguro e Suhai. O chatbot utiliza técnicas de processamento de linguagem natural para fornecer respostas precisas e úteis aos usuários.

## Instalação
Para instalar e executar o projeto localmente, siga os passos abaixo:

1. Instale o Ollama na sua máquina: [Ollama Download](https://ollama.com/download)
2. Faça pull do modelo usado como base para o chatbot, oráculo e "usuário":
    ```bash
    ollama pull mistral
    ```
3. Clone o repositório:
    ```bash
    git clone https://github.com/joaop/chatbot-seguro-automoveis.git
    ```
4. Navegue até o diretório do projeto:
    ```bash
    cd chatbot-seguro-automoveis
    ```
5. Crie e ative um ambiente virtual:
    ```bash
    conda create --name chatbot-env python=3.9
    conda activate chatbot-env
    ```
6. Instale as dependências:
    ```bash
    pip install pypdf==5.1.0 chromadb==0.5.23 sentence-transformers==3.2.1 ipywidgets==8.1.5 ollama==0.4.4 --no-cache-dir
    ```

## Uso
Para iniciar o chatbot, execute o comando:

## Estrutura do Projeto
- `src/chromadataset.py`: Contém a classe `ChromaDataset` responsável por gerenciar o banco de dados vetorial e processar PDFs.
- `src/chatbotseguros.py`: Contém a classe `ChatbotSeguros` que implementa a lógica do chatbot.
- `chatbottest.py`: Script para testar o chatbot e gerar relatórios de desempenho.