import random
import json
import uuid
import ollama
import numpy as np
from datetime import datetime
from src.chatbotseguros import ChatbotSeguros

class ChatbotTester:
    def __init__(self):
        self.model_name = 'mistral:latest'
        
        # perguntas base por categoria que servirão de exemplo para o Ollama
        self.perguntas_base = {
            'coberturas_protecoes': [
                "O seguro para o carro dá direito a carro reserva?",
                "Qual a extensão territorial do seguro para o carro?",
                "Seguro cobre um terceiro que bateu meu carro?", 
                "O seguro cobre roubo de acessórios instalados após a compra do carro?",
                "Posso escolher em qual oficina fazer o reparo?",
                "Se alguém riscar meu carro de propósito, o seguro cobre?",
                "O seguro cobre danos causados por enchentes ou alagamentos?",
                "O que acontece se meu carro for roubado com objetos pessoais dentro?",
                "O seguro cobre panes mecânicas ou elétricas?",
                "Se eu bater sozinho (sem envolver outros carros), o seguro cobre?",
                "O que é cobertura contra terceiros e o que ela inclui exatamente?",
                "Quem bate o carro alcoolizado tem direito à cobertura?",
                "Sou indenizado após uma árvore cair no meu carro?",
                "Meu carro foi danificado devido a um buraco na rua. O seguro cobre?"
            ],
            
            'servicos_beneficios': [
                "O seguro oferece serviço de chaveiro?",
                "Em caso de pane, posso solicitar reboque para qualquer lugar?",
                "Se acabar a bateria ou gasolina, a seguradora ajuda?",
                "Quantos quilômetros de guincho tenho direito?",
                "A seguradora oferece desconto para instalar rastreador?",
                "Posso incluir mais de um condutor no seguro?",
                "O que acontece se outra pessoa estiver dirigindo meu carro na hora do acidente?",
                "Existem limites para uso de assistência?",
                "Se eu contratar o carro reserva, tenho direito a utilizá-lo?",
                "Posso parcelar a franquia do seguro?"
            ],
            
            'pagamentos_valores': [
                "O valor do seguro muda se eu mudar de endereço?",
                "Posso pagar o seguro em mais de 12 vezes?",
                "Se eu vender o carro, posso transferir o seguro para o novo dono?",
                "O que é Tabela FIPE e como ela afeta o valor do seguro?",
                "Se eu fizer upgrade no carro (rodas, som), o valor do seguro muda?",
                "Existe desconto para bom condutor?",
                "O valor do seguro aumenta se eu usar muito a assistência?",
                "Quanto custa em média um seguro?",
                "Como é calculado o valor do seguro?",
                "Consigo contratar um seguro com nome sujo?"
            ],
            
            'processos_procedimentos': [
                "Preciso fazer vistoria para renovar o seguro?",
                "Como funciona a vistoria prévia?",
                "Se eu atrasar o pagamento, perco a cobertura imediatamente?",
                "Em caso de acidente, preciso esperar o perito para remover o carro?",
                "Como faço para incluir um reboque ou carretinha no seguro?",
                "Se eu trocar de carro durante a vigência, como faço?",
                "Quanto tempo tenho para comunicar um sinistro?",
                "Como faço para acionar a cobertura?",
                "O que devo fazer em caso de sinistro?",
                "O que é sinistro?"
            ],
            
            'casos_especificos': [
                "O seguro cobre se eu usar o carro para aplicativo (Uber/99)?",
                "Posso usar o carro em outros países?",
                "O que acontece se eu precisar trocar o para-brisa?",
                "Se eu modificar o carro (rebaixar, turbinar), perco a cobertura?",
                "O seguro cobre danos no carro se eu passar por uma manifestação?",
                "Se eu emprestar o carro para alguém e acontecer um acidente, estou coberto?",
                "O que acontece se eu me envolver em um acidente durante uma viagem?",
                "Meu carro sofreu danos devido a tumultos generalizados ou calamidade pública. Tenho direito à indenização?",
                "O seguro me indenizará se estiver com documento do carro irregular e IPVA atrasado?"
            ],
            
            'contratacao_renovacao': [
                "Preciso fazer nova vistoria se trocar de seguradora?",
                "Como faço para aproveitar o bônus da seguradora anterior?",
                "Posso contratar seguro para carro financiado?",
                "Existe idade máxima para o carro ter seguro?",
                "Quais documentos preciso para fazer o seguro?",
                "O que é perfil de condutor e como isso afeta meu seguro?",
                "Se eu tiver mais de um carro, ganho desconto?",
                "Qualquer carro pode ter seguro?",
                "Como escolher as coberturas do seguro para o carro?",
                "Consigo transferir o bônus que ganhei caso opte por mudar de seguradora?"
            ]
        }

        self.session_id = str(uuid.uuid4())
        self.resultados_teste = {
            'conversas': [],
            'metricas': {
                'duvidas': None,
                'assertividade': None
            }
        }

    def gerar_variacao_pergunta(self, pergunta_base):
        """
        Gera variação da pergunta com 60% de chance, caso contrário usa a pergunta original.
        """
        # 60% de chance de variar a pergunta
        if random.random() < 0.6:
            prompt = f"""
            Você é um cliente interessado em seguro de carro. 
            Reescreva a seguinte pergunta de uma maneira diferente, mantendo o mesmo significado 
            mas usando outras palavras, como se fosse uma pessoa real perguntando naturalmente:

            Pergunta original: {pergunta_base}

            Lembre-se:
            - Mantenha informal e natural
            - Pode adicionar contexto pessoal
            - Mantenha o foco na mesma dúvida
            - Use linguagem do dia a dia
            - Responda APENAS a nova pergunta, sem explicações

            Nova pergunta:"""

            try:
                resposta = ollama.chat(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                nova_pergunta = resposta['message']['content'].strip()
                # Remove aspas se houver
                nova_pergunta = nova_pergunta.strip('"\'')
                return nova_pergunta
            except Exception as e:
                print(f"Erro ao gerar variação da pergunta: {e}")
                return pergunta_base
        
        return pergunta_base

    def gerar_pergunta_followup(self, categoria, contexto_anterior):
        """Gera uma pergunta de follow-up baseada no contexto"""
        prompt = f"""
        Você é um cliente conversando sobre seguro de carro.
        Com base no contexto da conversa anterior, gere uma pergunta de follow-up natural.

        Contexto anterior: {contexto_anterior}

        Regras para a pergunta:
        - Deve ser relacionada ao mesmo assunto
        - Deve buscar mais detalhes ou esclarecimentos
        - Deve parecer natural e espontânea
        - Use linguagem informal
        - Responda APENAS a pergunta, sem explicações

        Pergunta de follow-up:"""

        try:
            resposta = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return resposta['message']['content'].strip().strip('"\'')
        except Exception as e:
            print(f"Erro ao gerar pergunta de follow-up: {e}")
            return f"E como funciona isso que você me explicou sobre {categoria}?"

    def gerar_conversa(self, chatbot, num_interacoes=5):
        historico = []
        categoria = random.choice(list(self.perguntas_base.keys()))
        
        for i in range(num_interacoes):
            if i == 0:
                # Primeira pergunta: variação de uma pergunta base
                pergunta_base = random.choice(self.perguntas_base[categoria])
                pergunta = self.gerar_variacao_pergunta(pergunta_base)
            else:
                # 70% de chance de fazer pergunta de follow up
                if random.random() < 0.7 and historico:
                    pergunta = self.gerar_pergunta_followup(
                        categoria, 
                        historico[-1]['pergunta'] + " | " + historico[-1]['resposta']
                    )
                else:
                    # Nova categoria e nova variação de pergunta
                    nova_categoria = random.choice(list(self.perguntas_base.keys()))
                    pergunta_base = random.choice(self.perguntas_base[nova_categoria])
                    pergunta = self.gerar_variacao_pergunta(pergunta_base)
            
            resposta = chatbot.gerar_resposta(pergunta, self.session_id)
            
            print(f"Usuário: {pergunta}")
            print(f"Chatbot: {resposta}")
            print('----')   
            
            historico.append({
                'timestamp': datetime.now().isoformat(),
                'pergunta': pergunta,
                'resposta': resposta
            })
            
        return historico

    def executar_testes(self, chatbot, num_conversas=3):
            """Same as original but stores metrics in a more structured way"""
            for i in range(num_conversas):
                conversa = self.gerar_conversa(chatbot, np.random.randint(1, 10))
                self.resultados_teste['conversas'].append({
                    'id_conversa': str(uuid.uuid4()),
                    'id_sessao': self.session_id,
                    'interacoes': conversa
                })
                
                self.session_id = str(uuid.uuid4())
            
            # Get metrics and format them for better visualization
            metricas_duvidas = chatbot.dataset_class.get_metricas_duvidas()
            metricas_assertividade = chatbot.dataset_class.get_metricas_assertividade()
            
            self.resultados_teste['metricas'] = {
                'duvidas': metricas_duvidas,
                'assertividade': metricas_assertividade
            }
            
            return self.resultados_teste

    def salvar_relatorio(self):
        # Save JSON data for potential future use
        filepath = f'resultados_teste_{self.session_id}.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            print(f'Salvando resultados do teste em {filepath}')
            json.dump(self.resultados_teste, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    chatbot = ChatbotSeguros()
    tester = ChatbotTester()
    
    resultados = tester.executar_testes(chatbot, num_conversas=25)
    tester.salvar_relatorio()