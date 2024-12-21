# chatbot-seguro-automoveis
 Repositório para armazenar chatbot capaz de responder dúvidas sobre seguros de automóveis das seguradoras Itaú, Bradesco, Porto Seguro e Suhai.

#

```
conda create --name chatbot-env python=3.9
conda activate chatbot-env
pip install pypdf==5.1.0 chromadb==0.5.23 sentence-transformers==3.2.1 ipywidgets==8.1.5 ollama==0.4.4 --no-cache-dir
```

ollama pull splitpierre/bode-alpaca-pt-br
ollama pull mistral
ollama pull tinyllama