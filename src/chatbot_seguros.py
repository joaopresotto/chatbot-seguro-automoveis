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

    def classificar_duvida(self, query):
        """
        Classifica o tipo de dúvida com base em padrões predefinidos.
        Retorna o tipo da dúvida e um score de confiança.
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
                    confidence = min(matches * 0.3, 1.0)  # Score máximo de 1.0
                    return duvida_type, confidence
        
        return 'outras_duvidas', 0.1
        
    def classificar_tema(self, query):
        """Classifica a pergunta em um tema específico"""
        temas = {
            'coberturas_protecoes': ['cobertura', 'cobre', 'protege', 'proteção', 'carro reserva', 'terceiro', 'roubo', 'acessórios', 'oficina', 'riscar', 'enchentes', 'alagamentos', 'objetos pessoais', 'panes mecânicas', 'panes elétricas', 'bater sozinho', 'cobertura contra terceiros', 'alcoolizado', 'árvore', 'buraco'],
            'servicos_beneficios': ['serviço de chaveiro', 'reboque', 'bateria', 'gasolina', 'quilômetros de guincho', 'rastreador', 'condutor', 'assistência', 'carro reserva', 'parcelar a franquia'],
            'pagamentos_valores': ['valor', 'preço', 'custo', 'pagamento', 'parcela', 'endereço', 'transferir seguro', 'Tabela FIPE', 'upgrade no carro', 'desconto', 'bom condutor', 'assistência', 'nome sujo'],
            'processos_procedimentos': ['vistoria', 'vistoria prévia', 'atrasar pagamento', 'perito', 'reboque', 'carretinha', 'comunicar sinistro', 'acionar cobertura', 'sinistro'],
            'casos_especificos': ['aplicativo', 'outros países', 'para-brisa', 'modificar carro', 'manifestação', 'emprestar carro', 'viagem', 'tumultos', 'calamidade pública', 'documento irregular', 'IPVA atrasado'],
            'contratacao_renovacao': ['nova vistoria', 'bônus', 'carro financiado', 'idade máxima', 'documentos', 'perfil de condutor', 'mais de um carro', 'qualquer carro', 'escolher coberturas', 'transferir bônus']
        }
        
        query_lower = query.lower()
        for tema, palavras_chave in temas.items():
            if any(palavra in query_lower for palavra in palavras_chave):
                return tema
        return 'outros'

    def avaliar_resposta(self, query, resposta, contexto):
        """
        Avalia a qualidade e assertividade da resposta.
        """
        try:
            # Prompt para avaliação
            eval_prompt = f"""
            Avalie a qualidade da resposta fornecida pelo chatbot com base nos seguintes critérios:

            1. **Texto alinhado ao tema**: A resposta fornecida está alinhada ao tema ou pergunta feita? Verifique se o chatbot se mantém dentro do escopo esperado (seguros de automóveis das seguradoras Santander, Bradesco, Porto Seguro e Suhai) ou se diverge para outros tópicos.
            2. **Texto preciso**: A resposta contém informações corretas, claras e relevantes? Evite considerar como precisas respostas que contenham informações erradas, confusas ou ambíguas.
            3. **Texto estruturado**: A resposta está bem estruturada e compreensível? Avalie se a linguagem é direta e acessível, sem ambiguidades ou termos desnecessariamente técnicos.
            4. **Texto no mesmo idioma**: O chatbot precisa responder as dúvidas sempre em português brasileiro, se a resposta estiver em outro idioma (inglês por exemplo), isso é um erro.
            5. **Texto no escopo**: Verifique se o chatbot evita responder perguntas fora de sua especialidade. Por exemplo, ele não deve tentar responder sobre seguros de saúde ou outros produtos financeiros que não sejam seguros de automóveis.

            Pergunta: {query}
            Resposta: {resposta}
            Contexto: {contexto}

            Retorne apenas um JSON com o seguinte formato exato:
            {{
                "texto_no_tema": true/false,
                "texto_preciso": true/false,
                "texto_estruturado": true/false,
                "texto_no_mesmo_idioma": true/false,
                "texto_no_escopo": true/false,
                "score": 0-100,
                "feedback": "Texto explicativo em português detalhando os pontos fortes e fracos da resposta."
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
                required_keys = ['texto_no_tema', 'texto_preciso', 'texto_estruturado', 'texto_no_mesmo_idioma', 'texto_no_escopo', 'score', 'feedback']
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
                "texto_estruturado": True,
                "texto_no_mesmo_idioma": True,
                "texto_no_escopo": True,
                'score': 50,
                'feedback': 'Avaliação padrão devido a erro no processo.'
            }

    def gerar_resposta(self, query, session_id):
        """
        Gera uma resposta para a query do usuário, registrando todo o processo.
        """
        # Classificar tipo de dúvida e registrar no dataset
        tipo_duvida, conf_duvida = self.classificar_duvida(query)
        self.dataset_class.registrar_duvida(query, tipo_duvida, conf_duvida, session_id)
        
        # Avaliar tema da pergunta
        tema = self.classificar_tema(query)
        
        # Buscar contexto relevante
        contexto_pdfs = self.dataset_class.busca_contextual(query)
        
        # Buscar conversas similares anteriores
        conversas_similares = self.dataset_class.buscar_conversas_similares(query)
        
        # Incluir histórico recente da sessão
        historico_recente = self.dataset_class.get_recent_history(session_id)
        
        prompt = f"""
        Você é um assistente especializado em seguros das seguradoras Santander, Bradesco, Porto Seguro e Suhai. Sua função é responder às perguntas dos usuários com base em informações fornecidas, mantendo precisão, profissionalismo e um tom acolhedor. 

        As informações a seguir são organizadas para ajudar na formulação de sua resposta:
        1. **Histórico recente da conversa**:
        {historico_recente}

        2. **Contexto dos documentos relevantes**:
        {contexto_pdfs}

        3. **Perguntas semelhantes respondidas anteriormente**:
        {conversas_similares}

        Responda à seguinte pergunta com base nas informações fornecidas, seguindo as diretrizes estabelecidas. Não inicie sua resposta com "Pergunta:" ou "Resposta:".  

        **Pergunta**: {query}
        """
        
        # Gerar resposta
        resposta = ollama.chat(
            model=self.model_chatbot,
            messages=[
                {'role': 'system', 'content': self.chatbot_content},
                {'role': 'user', 'content': prompt}
            ]
        )
        
        resposta_content = resposta['message']['content']
        
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