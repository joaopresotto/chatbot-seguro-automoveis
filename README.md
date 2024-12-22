# Chatbot Seguro Automóveis

## Descrição
Este projeto é um chatbot desenvolvido para responder dúvidas sobre seguros de automóveis das seguradoras Santander, Bradesco, Porto Seguro e Suhai. O chatbot utiliza técnicas de processamento de linguagem natural para fornecer respostas precisas e úteis aos usuários.

## Instalação
Para instalar e executar o projeto localmente:

1. Instale o [Ollama](https://ollama.com/download) e faça pull do modelo base:
    ```bash
    ollama pull mistral
    ```
2. Clone e configure o ambiente:
    ```bash
    git clone https://github.com/joaop/chatbot-seguro-automoveis.git
    cd chatbot-seguro-automoveis
    conda create --name chatbot-env python=3.9
    conda activate chatbot-env
    pip install pypdf==5.1.0 chromadb==0.5.23 sentence-transformers==3.2.1 ollama==0.4.4 streamlit==1.41.1 --no-cache-dir
    ```

## Estrutura do Projeto
- `src/chroma_dataset.py`: Contém a classe `ChromaDataset` responsável por gerenciar o banco de dados vetorial e processar PDFs.
- `src/chatbot_seguros.py`: Contém a classe `ChatbotSeguros` que implementa a lógica do chatbot.
- `chatbot_test.py`: Script para testar o chatbot e gerar relatórios de desempenho.

## Informações sobre o Projeto
O sistema é composto por três chatbots, cada um com função específica:

1. **Assistente de Seguros** (`data/chatbot_description.txt`): Chatbot principal que responde dúvidas sobre seguros.

2. **Avaliador Oráculo** (`data/oraculo_description.txt`): Avalia as respostas do assistente segundo:
   - `Texto alinhado ao tema`: Relevância à pergunta
   - `Texto preciso`: Clareza e ausência de ambiguidade
   - `Texto no mesmo idioma`: Consistência no português
   - `Texto no escopo`: Aderência à especialidade
   O oráculo gera um score (0-100) e feedback para cada resposta.

3. **Usuário Simulado**: Gera perguntas pré-definidas e follow-ups para testes.

### Categorias de Perguntas
- `coberturas_protecoes`: Eventos cobertos (roubos, acidentes, etc)
- `servicos_beneficios`: Assistência 24h, reboque, carro reserva
- `pagamentos_valores`: Custos e formas de pagamento
- `processos_procedimentos`: Vistoria, sinistros, manutenção
- `casos_especificos`: Uso comercial, viagens, modificações
- `contratacao_renovacao`: Contratação e renovação de apólices

Estas categorias são usadas tanto para geração de perguntas quanto para classificação das mesmas na classe `ChatbotSeguros`.

### Diagrama de Funcionamento
![Diagrama Chatbot](https://github.com/user-attachments/assets/e0aad4a4-f4c7-49d7-9b97-fbb7bd7de737)

### Avaliando o Chatbot
O script `chatbot_test.py` simula conversas e gera relatórios de desempenho. Para executar:
```bash
python chatbot_test.py
```

O [relatório gerado](https://github.com/joaopresotto/chatbot-seguro-automoveis/blob/main/resultados/resultados_teste_20241221_160629.json) contém:

1. `conversas`: Histórico completo dos testes
2. `metricas`: 
   - Classificação de temas
   - Avaliação do oráculo (scores e feedbacks)
   - Distribuição de temas e dúvidas frequentes

Exemplos de métricas geradas:
```json
"duvidas_frequentes": [
    [
      "Existem limites para uso de assistência?",2
    ],
    [
      "Preciso fazer vistoria para renovar o seguro?",2
    ],
    [
      "O seguro me indenizará se estiver com documento do carro irregular e IPVA atrasado?",2
    ],
    ...
]
```
```json
"temas_populares": {
    "coberturas_protecoes": 21,
    "outros": 106,
    "servicos_beneficios": 4,
    "pagamentos_valores": 5,
    "contratacao_renovacao": 1,
    "processos_procedimentos": 2
},
"media_score": 88.70503597122303,
"distribuicao_scores": {
    "excelente (90-100)": 100,
    "bom (70-89)": 34,
    "regular (50-69)": 5,
    "ruim (<50)": 0
}
```
```json
"problemas_detectados": [
    {
      "pergunta": "Então, vou deixar o meu carro para um amigo, se ele tiver um acidente, eu estou coberto, não?",
      "resposta": " Sim, caso você esteja emprestando seu carro a outra pessoa e houver um acidente durante o período de empréstimo, é possível que você seja coberto, mas isso dependerá das condições específicas da sua política de seguro automóvel. É necessário consultar as Condições Gerais específicas da sua seguradora para confirmar se você está realmente coberto nesse caso.",
      "timestamp": "2024-12-21T16:01:53.162513",
      "oraculo": {
        "tema": "outros",
        "avaliacao_score": 70,
        "texto_no_tema": true,
        "texto_preciso": false,
        "texto_no_mesmo_idioma": true,
        "texto_no_escopo": true,
        "feedback": "A resposta do chatbot está principalmente alinhada com os critérios apresentados (está no tópico, no mesmo idioma e no escopo). Porém, o texto é pouco preciso e falta um pouco de detalhamento. É possível explicitar mais sobre os pontos fortes e fracos da resposta para ajudar a melhorar a qualidade do serviço."
      }
    },
...
]
```
**Observação:** O avaliador oráculo, sendo também um chatbot, pode apresentar inconsistências em suas avaliações.

### Interface Gráfica
Para iniciar o chatbot, execute o comando:
<todo>

## Pontos de Melhoria
- Extrair chunks dos PDFs ao passar um modelo mais robusto, capaz de identificar parágrafos por exemplo.
    - Abordagem atual em `process_pdfs` extrai o texto da página como um todo e usa esse texto completo na parte de embedding
- Aprimorar deteção de temas, expandindo classes, possivelmente implementando um classificador e não comparação de palavras-chave.
- Aprimorar modelo oráculo, que muitas vezes avalia mal os problemas e dá *feedbacks* imprecisos.
