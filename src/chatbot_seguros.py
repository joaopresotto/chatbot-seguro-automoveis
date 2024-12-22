import os
import re
import json
import ollama
from src.chroma_dataset import ChromaDataset

class ChatbotSeguros:
    def __init__(self, database_path=os.path.join('data','chroma_database'), 
                 pdfs_path=os.path.join('data','docs')):
        """
        Inicializa o chatbot com suporte a histórico de conversas e análise de dúvidas.
        """
        # Configuração básica
        self.model_chatbot = 'mistral:latest'
        self.model_oraculo = 'mistral:latest' 
        
        # Carregar descrição do chatbot
        with open(os.path.join('data', 'chatbot_description.txt'), 'r', encoding='latin-1') as f:
            self.chatbot_content = f.read().encode('latin1').decode('utf-8')
            
        # Carregar descrição do oraculo
        with open(os.path.join('data', 'oraculo_description.txt'), 'r', encoding='latin-1') as f:
            self.oraculo_content = f.read().encode('latin1').decode('utf-8')
            
        # Instanciar dataset
        self.dataset_class = ChromaDataset(database_path=database_path, pdfs_path=pdfs_path)
        
        # Messages
        self.messages_chatbot = [
            # Mensagem inicial do sistema
            {'role': 'system', 'content': self.chatbot_content}
        ]

    def classificar_tema(self, query):
        """
        Classifica o tipo de tema com base em padrões predefinidos.
        Retorna o tipo da tema e um score de confiança.
        """
        question_patterns = {
            'coberturas_protecoes': [
                r'(?:o que|quais|qual).+(?:cobre|cobertura|coberto)',
                r'(?:tenho|tem) direito.+(?:cobertura|indenização)',
                r'(?:seguro|cobertura).+(?:carro reserva|terceiro)',
                r'(?:indeniza|indenizado|indenização).+(?:caso|quando|se)',
                r'(?:extensão|território|territorial).+(?:seguro|cobertura)',
                r'(?:proteção|protege|protegido).+(?:contra|para|em)',
                r'(?:roubo|furto|colisão|batida).+(?:cobre|cobertura)',
                r'(?:cobre|cobertura).+(?:roubo|furto|colisão|batida)',
                r'(?:panes mecânicas|panes elétricas|bater sozinho|riscar|enchentes|alagamentos|objetos pessoais).+(?:cobre|cobertura)',
                r'(?:alcoolizado|bebida).+(?:cobertura|direito)',
                r'(?:árvore|buraco|tumulto|calamidade).+(?:cobre|indeniza)',
            ],
            'servicos_beneficios': [
                r'(?:serviço de chaveiro|reboque|bateria|gasolina|quilômetros de guincho|rastreador|condutor|assistência|carro reserva|parcelar a franquia).+(?:como|funciona|usar|direito|oferece|solicitar|incluir|limite|vezes|usos)',
            ],
            'pagamentos_valores': [
                r'(?:valor|preço|custo|pagamento|parcela|endereço|transferir seguro|Tabela FIPE|upgrade no carro|desconto|bom condutor|assistência|nome sujo).+(?:muda|pago|pagar|afeta|aumenta|custa|calcula|calculado|contratar|fazer)',
            ],
            'processos_procedimentos': [
                r'(?:vistoria|vistoria prévia|atrasar pagamento|perito|reboque|carretinha|comunicar sinistro|acionar cobertura|sinistro).+(?:como|qual|preciso|necessário|fazer|acontece|tempo|prazo|devo|tenho|significa)',
            ],
            'casos_especificos': [
                r'(?:aplicativo|outros países|para-brisa|modificar carro|manifestação|emprestar carro|viagem|tumultos|calamidade pública|documento irregular|IPVA atrasado).+(?:cobre|indeniza|direito|acontece|perco|cobertura|seguro)',
            ],
            'contratacao_renovacao': [
                r'(?:nova vistoria|bônus|carro financiado|idade máxima|documentos|perfil de condutor|mais de um carro|qualquer carro|escolher coberturas|transferir bônus).+(?:como|preciso|necessário|fazer|contratar|seguro|cobertura|aproveitar|ganho|desconto)',
            ]
        }
        
        query_lower = query.lower()
        
        for duvida_type, patterns in question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    # Calcular score baseado na quantidade de matches
                    matches = len(re.findall(pattern, query_lower))
                    # confidence = min(matches * 0.3, 1.0)  # Score máximo de 1.0
                    return duvida_type
        
        return 'outros'

    def avaliar_resposta(self, query, resposta, contexto):
        """
        Avalia a qualidade e assertividade da resposta.
        """
        try:
            # Prompt para avaliação
            eval_prompt = f"""
            BASE DE CONHECIMENTO:
            Contexto: {contexto}
            Pergunta: {query}
            Resposta: {resposta}
            
            Avalie a qualidade da resposta fornecida pelo chatbot com base nos seguintes critérios:
            1. **Texto alinhado ao tema**: A resposta está diretamente relacionada à pergunta ou tema abordado? Verifique se o chatbot mantém o foco nas questões sobre *seguros de automóveis* das seguradoras Santander, Bradesco, Porto Seguro e Suhai, evitando desviar para outros tópicos.
            2. **Texto preciso**: A resposta é correta, clara e relevante? Identifique se há erros factuais, informações confusas ou ambiguidades que possam comprometer a utilidade da resposta.
            3. **Texto no mesmo idioma**: O chatbot deve sempre responder em português brasileiro (pt-br). Qualquer resposta total ou parcial em outro idioma (como inglês ou espanhol) será considerada um erro. Para este critério, avalie com atenção:
            - Se há uso integral de outro idioma.
            - Se há misturas entre português e outro idioma (mesmo em partes pequenas).
            - Se a linguagem utilizada segue os padrões de português brasileiro, considerando possíveis regionalismos ou influências externas.
            4. **Texto no escopo**: O chatbot não deve fornecer respostas fora de sua área de especialidade. Avalie se ele evita responder perguntas relacionadas a seguros de saúde ou outros produtos financeiros que não sejam seguros de automóveis.

            Retorne apenas um JSON com o seguinte formato exato:
            {{
                "texto_no_tema": true/false,
                "texto_preciso": true/false,
                "texto_no_mesmo_idioma": true/false,
                "texto_no_escopo": true/false,
                "score": (número inteiro de 0 a 100 representando o quão alinhada a resposta está com os critérios apresentados anteriormente),
                "feedback": "Texto explicativo em português detalhando os pontos fortes e fracos da resposta do chatbot."
            }}
            """
            
            # Avaliar resposta
            avaliacao = ollama.chat(
                model=self.model_oraculo,
                messages=[
                    {'role': 'system', 'content': self.oraculo_content},
                    {'role': 'user', 'content': eval_prompt}
                ]
            )
            
            # Processar resposta como JSON
            try:
                resultado = json.loads(avaliacao['message']['content'])
                # Garantir que todas as chaves necessárias existem
                required_keys = ['texto_no_tema', 'texto_preciso', 'texto_no_mesmo_idioma', 'texto_no_escopo', 'score', 'feedback']
                for key in required_keys:
                    if key not in resultado:
                        raise KeyError(f"Chave '{key}' ausente na avaliação")
                return resultado
            except json.JSONDecodeError:
                print("Erro ao decodificar JSON da avaliação")
                raise
            
        except Exception as e:
            print(f"Erro na avaliação: {e}")
            # Retornar resultado padrão em caso de erro
            return {
                "texto_no_tema": True,
                "texto_preciso": True,
                "texto_no_mesmo_idioma": True,
                "texto_no_escopo": True,
                'score': 50,
                'feedback': 'Avaliação padrão devido a erro no processo.'
            }

    def gerar_resposta(self, query, session_id='default'):
        """
        Gera uma resposta para a query do usuário, registrando todo o processo.
        """
        # avaliar tema da pergunta
        tema = self.classificar_tema(query)
        
        # registrar pergunta
        self.dataset_class.registrar_duvida(query, tema, session_id)
        
        # Buscar contexto relevante
        contexto_pdfs = self.dataset_class.busca_contextual(query)
        
        # Buscar conversas similares anteriores
        conversas_similares = self.dataset_class.buscar_conversas_similares(query)
        
        prompt = f"""
        Você é um assistente especializado EXCLUSIVAMENTE em seguros de automóveis (AUTO) das seguradoras Santander, Bradesco, Porto Seguro e Suhai.

        REGRAS ABSOLUTAS:
        1. RESPONDA APENAS O QUE FOI PERGUNTADO
        2. NÃO FAÇA suposições ou deduções
        3. NÃO MENCIONE "contexto", "documentos" ou "informações fornecidas"
        4. NÃO USE frases introdutórias como "vou responder", "a seguir", etc
        5. NÃO FAÇA perguntas adicionais não solicitadas
        6. NÃO CITE informações técnicas sem uma pergunta específica

        TIPOS DE INTERAÇÃO:
        1. SAUDAÇÕES/MENSAGENS SIMPLES (oi, olá, bom dia, tudo bem, etc):
        RESPONDA APENAS: "Olá! Sou um assistente especializado em seguros auto da Santander, Bradesco, Porto Seguro e Suhai. Como posso ajudar?"
        2. PERGUNTAS FORA DO ESCOPO (não relacionadas a seguros auto das 4 seguradoras):
        RESPONDA APENAS: "Desculpe, só posso responder sobre seguros de automóveis das seguradoras Santander, Bradesco, Porto Seguro e Suhai."
        3. PERGUNTAS ESPECÍFICAS SOBRE SEGUROS AUTO:
        - Responda EXATAMENTE o que foi perguntado
        - Use APENAS informações relevantes à pergunta
        - Se não houver informação específica, RESPONDA APENAS: "Não tenho essa informação específica. Recomendo contatar diretamente a seguradora."

        DADOS PARA USO RESTRITO (usar SOMENTE para perguntas específicas sobre seguros auto):
        Contexto relacionado com a pergunta: {contexto_pdfs}
        Histórico de mensagens similares: {conversas_similares}
         
        Responda à seguinte pergunta com base nas informações fornecidas, seguindo as diretrizes estabelecidas anteriormente: 
        Pergunta: {query}
        """
        
        # Adiciona pergunta do usuário
        self.messages_chatbot.append({'role': 'user', 'content': prompt})
        
        # Gerar resposta
        resposta = ollama.chat(
            model=self.model_chatbot,
            messages=self.messages_chatbot,
        )
        resposta_content = resposta['message']['content']
        
        # Adiciona resposta do chatbot ao histórico
        self.messages_chatbot.append({'role': 'assistant', 'content': resposta_content})
        
        # Fix pois por conta do historico de conversas, a resposta pode começar com "Pergunta:" ou "Resposta:" por ser um modelo mais "simples"
        if resposta_content.strip().startswith("Pergunta:"):
            resposta_content = resposta_content.split("Pergunta:")[1].strip()
        elif resposta_content.strip().startswith("Resposta:"):
            resposta_content = resposta_content.split("Resposta:")[1].strip()
        
        # Avaliar resposta
        avaliacao = self.avaliar_resposta(query, resposta_content, contexto_pdfs)
        
        # Registrar interação
        self.dataset_class.registrar_interacao(session_id, query, resposta_content, tema, avaliacao)
        
        return resposta_content