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

## Estrutura do Projeto
- `src/chroma_dataset.py`: Contém a classe `ChromaDataset` responsável por gerenciar o banco de dados vetorial e processar PDFs.
- `src/chatbot_seguros.py`: Contém a classe `ChatbotSeguros` que implementa a lógica do chatbot.
- `chatbot_test.py`: Script para testar o chatbot e gerar relatórios de desempenho.

## Informações sobre o Projeto
Além do assistende de seguros, o projeto implementa outros dois chatbots com o intuito de gerar uma maneira mais simplificada para avaliação do assistente. A todo então temos três chatbots no projeto:
1. Assistente de Seguros (`data/chatbot_description.txt`)
2. Oráculo Avaliador (`data/oraculo_description.txt`) 
A ideia do oráculo é avaliar se as respostas fornecidas pelo assistente seguem algumas diretrizes essenciais:
- 1. `Texto alinhado ao tema`: verifica se a resposta está diretamente relacionada à pergunta ou tema abordado.
- 2. `Texto preciso`: verifica se a resposta é clara, se não possui informações confusas ou ambiguidade.
- 3. `Texto no mesmo idioma`: avalia se o chatbot mantêm coerência de idioma, mantendo-se sempre em português.
- 4. `Texto no escopo`: verifica se o chatbot responde somente sobre temas dentro da sua área de especialidade.
Essas informações tem como objetivo facilitar a avaliação dos resultados gerados com script de teste (`chatbot_test.py`), mas como dependem de outro chatbot para serem avaliadas podem estar sujeitas a erros de avaliação, muitas vezes não penalizando erros claros como trocas de idioma por exemplo.
3. "Usuário" 
No script de teste (`chatbot_test.py`), para podermos avaliar de maneira mais assertiva o chatbot foi também criado um terceiro chatbot capaz de replicar uma série de perguntas pré-definidas (com uma chance de gerar uma variação da mesma) e capaz também de gerar perguntas de *follow up*, seguindo a linha de raciocínio da primeira pergunta. 
Os temas de perguntas abordados são:
- `coberturas_protecoes`: Questões sobre eventos ou situações cobertas pelo seguro, como roubos, acidentes e danos a terceiros.  
- `servicos_beneficios`: Dúvidas sobre serviços e benefícios adicionais, como assistência 24h, reboque e carro reserva.  
- `pagamentos_valores`: Perguntas relacionadas a custos, formas de pagamento e fatores que impactam o valor do seguro.  
- `processos_procedimentos`: Instruções sobre processos administrativos, como vistoria, sinistros e manutenção da apólice.  
- `casos_especificos`: Situações particulares, como uso comercial do veículo, viagens internacionais e modificações no carro.  
- `contratacao_renovacao`: Informações sobre contratação inicial e renovação de apólices, como troca de seguradora e aproveitamento de bônus.  

### O que acontece quando uma pergunta é feita?
![Diagrama Chatbot](https://github.com/user-attachments/assets/e0aad4a4-f4c7-49d7-9b97-fbb7bd7de737)

- embedding da pergunta e com ele são feitas duas buscas:
    - a primeira busca no banco de dados de PDFs quais são as 10 páginas mais similares (informação de contexto)
    - a segunda busca no banco de dados recupera quais mensagens de todo o histórico são similares (histórico recente)
- 

é criado e feito uma busca de quais partes dos documentos são mais similares
2.

### Avaliando o chatbot
Com o script `chatbot_test.py` é possível simular diversas conversas com o chatbot com o intuito de gerar um relatório apresentando métricas de desempenho do assistente.
    ```bash
    python chatbot_test.py
    ```

### Interface gráfica
    Para iniciar o chatbot, execute o comando:
    <todo>

## Pontos de melhoria
- Extrair chunks dos PDFs ao passar um modelo mais robusto, capaz de identificar parágrafos por exemplo.
    - Abordagem atual em `process_pdfs` extrai o texto da página como um todo e usa esse texto completo na parte de embedding
