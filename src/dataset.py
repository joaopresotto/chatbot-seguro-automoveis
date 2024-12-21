import os
import chromadb
from datetime import datetime
from collections import defaultdict
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer # para fazer embedding dos pdfs e historico de conversa!

class ChromaDataset:
    def __init__(self, database_path=os.path.join('data','chroma_database'), pdfs_path=os.path.join('data','docs')):
        # Inicializar banco de dados vetorial
        self.client = chromadb.PersistentClient(path=database_path)
        
        # Coleções no ChromaDB
        self.docs_collection = self.client.get_or_create_collection(
            name="seguros_automoveis_embeddings"
        )
        self.history_collection = self.client.get_or_create_collection(
            name="chat_history"
        )
        self.questions_collection = self.client.get_or_create_collection(
            name="questions_analysis"
        )
    
        # Processar PDFs se necessário
        if self.docs_collection.count() == 0:
            print("Processando PDFs...")
            self.save_documents(self.docs_collection, self.process_pdfs(pdfs_path))
            
    def registrar_duvida(self, query, duvida_type, confidence, session_id) :
        """
        Registra a dúvida do usuário para análise posterior.
        """
        timestamp = datetime.now().isoformat()
        
        # Criar embedding da dúvida
        modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')
        question_embedding = modelo_embedding.encode(query).tolist()
        
        # Registrar no ChromaDB
        self.questions_collection.add(
            embeddings=[question_embedding],
            documents=[query],
            metadatas=[{
                'session_id': session_id,
                'timestamp': timestamp,
                'duvida_type': duvida_type,
                'confidence': confidence
            }],
            ids=[f"question_{timestamp}_{session_id}"]
        )
        
    def buscar_conversas_similares(self, query, top_k=5):
        """Busca conversas similares baseadas na query"""
        modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = modelo_embedding.encode(query).tolist()
        
        results = self.history_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return "\n".join(results['documents'][0])
    
    def get_recent_history(self, session_id, limit=5):
        """Get recent conversation history from ChromaDB"""
        # Pegamos todos os documentos da sessão
        results = self.history_collection.get(
            where={"session_id": session_id}
        )
        
        if results['documents']:
            # Organizamos por timestamp manualmente
            conversations = list(zip(
                results['documents'], 
                results['metadatas']
            ))
            
            # Ordenar por timestamp em ordem decrescente
            sorted_conversations = sorted(
                conversations,
                key=lambda x: x[1]['timestamp'],
                reverse=True
            )
            
            # Pegar apenas os últimos 'limit' documentos
            recent_conversations = sorted_conversations[:limit]
            
            # Retornar apenas os documentos, já ordenados
            return "\n".join(doc for doc, _ in recent_conversations)
        
        return ""
    
    def busca_contextual(self, query, top_k=10):
        # Gerar embedding da query
        # all-MiniLM-L6-v2 -> All-round model tuned for many use-cases. Trained on a large and diverse dataset of over 1 billion training pairs.
        modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')
        embedding_query = modelo_embedding.encode(query).tolist()
        
        # Busca por similaridade
        resultados = self.docs_collection.query(
            query_embeddings=[embedding_query],
            n_results=top_k
        )

        return "\n".join(resultados['documents'][0])
        
    def registrar_interacao(self, session_id, query, resposta, tema, avaliacao):
        """
        Registra a interação completa no histórico.
        """
        timestamp = datetime.now().isoformat()
        
        # Preparar documento para embedding
        documento = f"""
        Sessão: {session_id}
        Timestamp: {timestamp}
        Pergunta: {query}
        Resposta: {resposta}
        Tema: {tema}
        Avaliação Score: {avaliacao['score']}
        Feedback: {avaliacao['feedback']}
        """
        
        # Gerar embedding
        modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = modelo_embedding.encode(documento).tolist()
        
        # Salvar no ChromaDB
        self.history_collection.add(
            embeddings=[embedding],
            documents=[documento],
            metadatas=[{
                'session_id': session_id,
                'timestamp': timestamp,
                'query': query,
                'tema': tema,
                'avaliacao_score': avaliacao['score'],
                'relevante': avaliacao['relevante'],
                'preciso': avaliacao['preciso'],
                'claro': avaliacao['claro'],
                'feedback': avaliacao['feedback']
            }],
            ids=[f"chat_{timestamp}_{session_id}"]
        )

    def get_metricas_assertividade(self):
        """
        Retorna métricas sobre a assertividade das respostas.
        """
        # Obter todas as interações
        all_interactions = self.history_collection.get()
        
        total_interacoes = len(all_interactions['documents'])
        temas = defaultdict(int)
        scores = []
        problemas = []
        
        for metadata in all_interactions['metadatas']:
            scores.append(metadata['avaliacao_score'])
            temas[metadata['tema']] += 1 # acrescentar tema
            # Registrar problemas se score baixo
            if metadata['avaliacao_score'] < 70:
                problemas.append({
                    'query': metadata['query'],
                    'score': metadata['avaliacao_score'],
                    'timestamp': metadata['timestamp']
                })
        
        return {
            'total_interacoes': total_interacoes,
            'temas_populares': dict(temas),
            'media_score': sum(scores) / len(scores) if scores else 0,
            'problemas_detectados': problemas,
            'distribuicao_scores': {
                'excelente (90-100)': len([s for s in scores if s >= 90]),
                'bom (70-89)': len([s for s in scores if 70 <= s < 90]),
                'regular (50-69)': len([s for s in scores if 50 <= s < 70]),
                'ruim (<50)': len([s for s in scores if s < 50])
            }
        }
        
    def get_metricas_duvidas(self):
        """
        Retorna métricas sobre as dúvidas dos usuários.
        """
        # Obter todas as dúvidas registradas
        all_questions = self.questions_collection.get()
        
        # Análise das dúvidas
        total_duvidas = len(all_questions['documents'])
        duvidas_por_tipo = defaultdict(int)
        duvidas_comuns = defaultdict(int)
        
        for metadata in all_questions['metadatas']:
            duvidas_por_tipo[metadata['duvida_type']] += 1
        
        # Encontrar dúvidas similares
        for i, doc in enumerate(all_questions['documents']):
            duvidas_comuns[doc] += 1
        
        return {
            'total_duvidas': total_duvidas,
            'distribuicao_tipos': dict(duvidas_por_tipo),
            'duvidas_frequentes': sorted(
                duvidas_comuns.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }

    ## Passo 1
    # - extrair texto dos PDFs (pypdf)
    # - com os textos criar embeddings usando SentenceTransformer

    # Extrai textos do PDF da variável PATH usando a biblioteca pypdf
    # Retorna uma lista com as N páginas onde cada entrada é o texto extraído da página
    def extract_text_from_pdf(self, path):
        def clean_text(text):
            # Remove espaços extras e quebras de linha desnecessárias
            text = ' '.join(text.split())
            # Remove caracteres de controle que podem causar problemas
            text = ''.join(char for char in text if char.isprintable())
            return text.strip()
        
        with open(path, 'rb') as f:
            reader = PdfReader(f)
            extracted_text = []
            
            for page in reader.pages:
                text = page.extract_text()
                cleaned_text = clean_text(text)
                if cleaned_text:  # Só adiciona se tiver texto
                    extracted_text.append(cleaned_text)
                    
        return extracted_text

    # Função para criar embedding do texto extraído dos PDFs
    def process_pdfs(self, directory):
        # Cria instância do modelo
        # all-MiniLM-L6-v2 -> All-round model tuned for many use-cases. Trained on a large and diverse dataset of over 1 billion training pairs.
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
        
        # Percorre o diretório extraindo as informações de todos os arquivos
        processed_data = []
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                extracted_text = self.extract_text_from_pdf(os.path.join(directory, filename))   
                c_embedding = embedding_model.encode(extracted_text)      
                processed_data.append({
                    'filename': filename,
                    'extracted_text': extracted_text,
                    'embedding': c_embedding
                })
                
        return processed_data

    ## Passo 2
    # - armazenar detecções em banco de dados local

    def save_documents(self, collection, processed_data):
        # Salvar documentos
        for idx, doc in enumerate(processed_data):
            num_embeddings = len(doc['embedding'])
            num_documents = len(doc['extracted_text'])

            if num_embeddings != num_documents:
                raise ValueError(f"Embeddings e documentos têm tamanhos diferentes: {num_embeddings} != {num_documents}")

            # Gerar uma lista de IDs correspondente ao número de embeddings/documentos
            ids = [f"doc_{idx}_{i}" for i in range(num_embeddings)]
            
            # Filtrar IDs que já existem
            ids_to_add = [id_ for id_ in ids if id_ not in set(collection.get()['ids'])]

            # Se não houver novos IDs, pule para o próximo documento
            if not ids_to_add:
                print(f"Todos os documentos de 'doc_{idx}' já existem. Pulando...")
                continue

            # Gerar metadados, a extração é feita sob cada pagina dos pdfs, então vamos criar a info de acordo
            metadatas = [{'filename': doc['filename'], 'page': i+1} for i in range(num_embeddings)]

            collection.add(
                embeddings=doc['embedding'],
                metadatas=metadatas,
                documents=doc['extracted_text'],
                ids=ids
            )
            