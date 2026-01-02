import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from datetime import datetime
import requests
import json

# Load environment variables
load_dotenv()

class RAGChatbot:
    def __init__(self):
        # Initialize APIs
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        # Initialize embeddings and LLM
        self.embeddings = None
        self.llm = None
        self.vector_store = None
        self.qa_chain = None
        
        if self.google_api_key:
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=self.google_api_key
                )
                
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    google_api_key=self.google_api_key,
                    temperature=0.3
                )
                print("✓ Google Gemini AI initialized")
            except Exception as e:
                print(f"✗ Error initializing Google AI: {e}")
        
        if self.tavily_api_key:
            print("✓ Tavily web search available")
        else:
            print("✗ Tavily API key not found. Web search disabled.")
        
        self.initialize_rag_system()
    
    def initialize_rag_system(self):
        """Initialize the RAG system with knowledge base"""
        try:
            if not self.embeddings or not self.llm:
                print("⚠ Using simple mode (no vector store)")
                return
            
            # Load documents
            loader = TextLoader("knowledge_base/parallel_computing_docs.txt")
            documents = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                chunks,
                self.embeddings,
                persist_directory="./chroma_db"
            )
            self.vector_store.persist()
            
            # Create retrieval QA chain
            prompt_template = """
            You are an expert assistant for Parallel and Distributed Computing course.
            
            CONTEXT FROM COURSE MATERIALS:
            {context}
            
            ADDITIONAL WEB INFORMATION (if available):
            {web_info}
            
            USER QUESTION: {question}
            
            INSTRUCTIONS:
            1. First, answer based on the course materials context
            2. If web information is provided, use it to enhance your answer
            3. If the question asks for current/recent information, prioritize web information
            4. If web information contradicts course materials, mention both perspectives
            5. Always cite sources when possible
            6. Provide practical examples and applications
            
            Provide a detailed, educational answer suitable for computer science students.
            
            ANSWER:
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "web_info", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            print("✓ RAG system initialized successfully!")
            
        except Exception as e:
            print(f"✗ Error initializing RAG system: {e}")
            self.qa_chain = None
    
    def search_web_tavily(self, query):
        """Search the web using Tavily API"""
        if not self.tavily_api_key:
            return "Web search is not configured. Please add TAVILY_API_KEY to .env file."
        
        try:
            # Tavily API endpoint
            url = "https://api.tavily.com/search"
            
            # Prepare request
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
                "include_images": False,
                "include_answer": True,
                "include_raw_content": False
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # Make API call
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Format results
            web_info = "🌐 **WEB SEARCH RESULTS:**\n\n"
            
            # Include Tavily's answer if available
            if data.get('answer'):
                web_info += f"**AI Summary:** {data['answer']}\n\n"
            
            # Include search results
            if data.get('results'):
                web_info += "**Sources Found:**\n"
                for i, result in enumerate(data['results'][:3], 1):
                    web_info += f"{i}. **{result.get('title', 'Untitled')}**\n"
                    web_info += f"   📍 {result.get('url', 'No URL')}\n"
                    web_info += f"   📝 {result.get('content', 'No content available')[:250]}...\n\n"
            else:
                web_info += "No relevant web results found for this query.\n\n"
            
            web_info += f"*Search performed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
            
            return web_info
            
        except requests.exceptions.Timeout:
            return "⚠ Web search timed out. Please try again later."
        except requests.exceptions.RequestException as e:
            return f"⚠ Web search error: {str(e)}"
        except Exception as e:
            return f"⚠ Unexpected error in web search: {str(e)}"
    
    def query(self, question, use_web_search=False):
        """Process a user question with optional web search"""
        try:
            web_info = ""
            
            # Perform web search if requested and API key is available
            if use_web_search and self.tavily_api_key:
                print(f"🔍 Searching web for: {question}")
                web_info = self.search_web_tavily(question)
            
            if self.qa_chain:
                # Prepare context for LLM
                result = self.qa_chain({
                    "query": question,
                    "web_info": web_info if web_info else "No web search performed or available."
                })
                
                sources = []
                if result.get("source_documents"):
                    sources = [
                        {
                            "source": "Course Materials",
                            "content": doc.page_content[:200] + "..."
                        }
                        for doc in result.get("source_documents", [])
                    ]
                
                # Add web search info if used
                if use_web_search and web_info and "WEB SEARCH RESULTS:" in web_info:
                    sources.append({
                        "source": "Web Search",
                        "content": "Information retrieved from recent web sources"
                    })
                
                return {
                    "answer": result["result"],
                    "sources": sources,
                    "metadata": {
                        "used_web_search": use_web_search and bool(web_info),
                        "has_tavily_key": bool(self.tavily_api_key),
                        "timestamp": datetime.now().isoformat()
                    }
                }
            else:
                # Fallback: Simple response
                return self.get_fallback_response(question, use_web_search)
                
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "metadata": {
                    "used_web_search": False,
                    "has_tavily_key": bool(self.tavily_api_key),
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def get_fallback_response(self, question, use_web_search=False):
        """Provide a fallback response when RAG system is not available"""
        responses = {
            "parallel computing": "**Parallel Computing** uses multiple processors simultaneously to solve computational problems faster by dividing them into smaller tasks that can be executed concurrently.",
            
            "distributed computing": "**Distributed Computing** involves multiple computers working together over a network, each with its own memory, communicating via message passing protocols.",
            
            "mpi": "**MPI (Message Passing Interface)** is a standardized message-passing system for parallel computing in distributed memory systems. It provides point-to-point and collective communication operations.",
            
            "openmp": "**OpenMP** is an API for shared memory parallel programming using compiler directives to parallelize loops and sections of code, making it easier to add parallelism to existing sequential code.",
            
            "cuda": "**CUDA** is NVIDIA's parallel computing platform for GPU programming, enabling general-purpose computing on GPUs (GPGPU). It's widely used for scientific computing and machine learning.",
            
            "amdahl's law": "**Amdahl's Law** gives the theoretical speedup limit when parallelizing a program: `Speedup = 1 / (S + (1-S)/N)` where S is the sequential fraction and N is number of processors.",
            
            "load balancing": "**Load Balancing** distributes work evenly across processors to optimize resource utilization, maximize throughput, and minimize response time in parallel systems.",
            
            "synchronization": "**Synchronization** coordinates concurrent processes using mechanisms like locks, semaphores, barriers, and monitors to ensure correct execution order.",
            
            "mapreduce": "**MapReduce** processes large datasets with parallel, distributed algorithms using Map (process/filter) and Reduce (aggregate) phases, popularized by Google.",
            
            "gpu": "**GPU Programming** utilizes Graphics Processing Units with thousands of cores optimized for parallel processing of graphics and general compute tasks.",
            
            "shared memory": "**Shared Memory Architectures** allow all processors to access the same memory space, simplifying programming but requiring synchronization.",
            
            "distributed memory": "**Distributed Memory Architectures** have each processor with its own memory, requiring explicit message passing for communication.",
            
            "quantum computing": "**Quantum Computing** uses quantum bits (qubits) that can exist in multiple states simultaneously, enabling massive parallel computation for specific problems."
        }
        
        # Default response
        answer = """🤖 **Welcome to Parallel Computing Assistant!**

I can help you understand concepts like:
• **Parallel Architectures**: Shared vs Distributed memory
• **Programming Models**: MPI, OpenMP, CUDA, MapReduce
• **Concepts**: Load balancing, Synchronization, Amdahl's Law
• **Applications**: Scientific computing, Machine learning, Big data

Try asking specific questions like:
• "What is the difference between MPI and OpenMP?"
• "How does CUDA work?"
• "Explain load balancing techniques"
• "What are recent trends in parallel computing?"
"""
        
        # Check for keywords
        question_lower = question.lower()
        for keyword, response in responses.items():
            if keyword in question_lower:
                answer = response
                break
        
        # Add web search note if requested but not available
        if use_web_search and not self.tavily_api_key:
            answer += "\n\n⚠ **Note:** Web search requires a Tavily API key. Add `TAVILY_API_KEY=your_key_here` to your `.env` file to enable web search."
        elif use_web_search:
            answer += "\n\n🔍 *Web search was performed to enhance this answer*"
        
        return {
            "answer": answer,
            "sources": [{"source": "Course Knowledge Base"}],
            "metadata": {
                "used_web_search": use_web_search,
                "has_tavily_key": bool(self.tavily_api_key),
                "timestamp": datetime.now().isoformat()
            }
        }

# Singleton instance
chatbot_instance = RAGChatbot()