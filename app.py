from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from datetime import datetime
import requests
import json

# Load environment variables
load_dotenv()

# ========== RAG CHATBOT CLASS ==========
class RAGChatbot:
    def __init__(self):
        # Initialize APIs
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        print("=" * 50)
        print("PARALLEL COMPUTING CHATBOT INITIALIZING...")
        print("=" * 50)
        
        # Check API keys
        if self.google_api_key:
            print("✓ Google Gemini API key found")
        else:
            print("✗ Google Gemini API key not found (add to .env file)")
        
        if self.tavily_api_key:
            print("✓ Tavily web search API key found")
        else:
            print("✗ Tavily API key not found (web search disabled)")
        
        # Enhanced knowledge base with better keyword matching
        self.knowledge_base = {
            "parallel computing": {
                "answer": "**Parallel Computing** uses multiple processors or cores simultaneously to solve computational problems faster by dividing them into smaller tasks that can be executed concurrently.\n\n**Key Points:**\n1. **Speedup**: S(p) = T(1) / T(p)\n2. **Efficiency**: E(p) = S(p) / p\n3. **Types**: Data parallelism, Task parallelism\n4. **Applications**: Scientific simulations, Weather forecasting, ML training",
                "keywords": ["parallel", "parallel computing", "parallelism", "multi-core", "speedup"],
                "examples": ["Scientific simulations", "Weather forecasting", "Machine learning training"]
            },
            "distributed computing": {
                "answer": "**Distributed Computing** involves multiple computers (nodes) working together over a network to solve a single problem. Each node has its own memory and communicates via message passing.\n\n**Key Points:**\n1. **Resource sharing** across network\n2. **Concurrency** of operations\n3. **Scalability** - add more nodes\n4. **Fault tolerance** - system continues if nodes fail",
                "keywords": ["distributed", "distributed computing", "distributed systems", "nodes", "cluster"],
                "examples": ["Cloud computing", "Blockchain networks", "Distributed databases"]
            },
            "mpi": {
                "answer": "**MPI (Message Passing Interface)** is a standardized message-passing system used for parallel computing in distributed memory architectures.\n\n**Key Functions:**\n- `MPI_Send` / `MPI_Recv` (point-to-point)\n- `MPI_Bcast` (broadcast)\n- `MPI_Reduce` (reduction)\n- `MPI_Barrier` (synchronization)\n\n**When to use:** Distributed memory systems, clusters, supercomputers",
                "keywords": ["mpi", "message passing", "message passing interface", "mpi programming"],
                "examples": ["High-performance computing clusters", "Scientific simulations", "Large-scale data processing"]
            },
            "openmp": {
                "answer": "**OpenMP (Open Multi-Processing)** is an API for shared memory parallel programming. It uses compiler directives (pragmas) to parallelize loops and sections of code.\n\n**Key Directives:**\n- `#pragma omp parallel` (creates parallel region)\n- `#pragma omp for` (parallelizes loop)\n- `#pragma omp sections` (task parallelism)\n\n**When to use:** Shared memory systems, multi-core CPUs, loops with no dependencies",
                "keywords": ["openmp", "open mp", "omp", "pragma", "shared memory"],
                "examples": ["Multi-core processors", "Symmetric multiprocessing systems", "Parallelizing loops in C/C++/Fortran"]
            },
            "cuda": {
                "answer": "**CUDA (Compute Unified Device Architecture)** is NVIDIA's parallel computing platform and programming model for GPUs.\n\n**Key Concepts:**\n- **Grid** → **Blocks** → **Threads** hierarchy\n- **Kernels**: Functions that run on GPU\n- **Memory Hierarchy**: Global, Shared, Local, Constant\n\n**Programming:** CUDA C/C++ extensions with `__global__`, `__device__`, `__host__` keywords",
                "keywords": ["cuda", "gpu", "gpu programming", "nvidia", "gpu computing"],
                "examples": ["Deep learning training", "Scientific computing", "Image and video processing"]
            },
            "amdahl's law": {
                "answer": "**Amdahl's Law** gives the theoretical maximum speedup achievable when parallelizing a program:\n\n`Speedup = 1 / (S + (1-S)/N)`\n\n**Where:**\n- **S** = Sequential fraction (portion that cannot be parallelized)\n- **N** = Number of processors\n\n**Example:** If 30% is sequential (S=0.3) with 4 processors (N=4):\n`Speedup = 1 / (0.3 + 0.7/4) = 2.1`\n\n**Implication:** Sequential portion limits max speedup",
                "keywords": ["amdahl", "amdahl's law", "speedup law", "parallel speedup", "sequential fraction"],
                "examples": ["Performance prediction", "Parallel system design", "Bottleneck analysis"]
            },
            "load balancing": {
                "answer": "**Load Balancing** distributes computational workload evenly across available processors to optimize resource utilization.\n\n**Types:**\n1. **Static**: Distribution decided at compile time (Block, Cyclic)\n2. **Dynamic**: Distribution adjusted at runtime (Work stealing)\n\n**Goals:** Maximize throughput, minimize response time, avoid idle processors",
                "keywords": ["load balancing", "load balance", "task distribution", "workload distribution"],
                "examples": ["Task scheduling", "Cloud computing", "Web server clusters"]
            },
            "synchronization": {
                "answer": "**Synchronization** coordinates the execution of multiple threads or processes in parallel computing.\n\n**Mechanisms:**\n1. **Locks/Mutexes** - Ensure mutual exclusion\n2. **Semaphores** - Control access to resources\n3. **Barriers** - Synchronize all threads\n4. **Monitors** - Encapsulate data and procedures\n\n**Problems:** Deadlock, Starvation, Race conditions",
                "keywords": ["synchronization", "synchronize", "locks", "mutex", "semaphore", "barrier"],
                "examples": ["Critical section protection", "Producer-consumer problems", "Parallel algorithm coordination"]
            },
            "mapreduce": {
                "answer": "**MapReduce** is a programming model for processing large datasets with parallel, distributed algorithm.\n\n**Phases:**\n1. **Map**: Processes input key-value pairs → intermediate pairs\n2. **Shuffle**: Groups intermediate values by key\n3. **Reduce**: Aggregates values for each key\n\n**Hadoop:** HDFS (storage) + MapReduce (processing) + YARN (resource management)",
                "keywords": ["mapreduce", "map reduce", "hadoop", "big data processing"],
                "examples": ["Big data processing", "Web indexing", "Log analysis"]
            },
            "shared memory": {
                "answer": "**Shared Memory Architecture** allows all processors to access a common memory space.\n\n**Types:**\n1. **UMA (Uniform Memory Access)**: Equal access time\n2. **NUMA (Non-Uniform Memory Access)**: Access time varies\n\n**Advantages:** Easy programming, fast communication\n**Disadvantages:** Memory contention, cache coherence issues",
                "keywords": ["shared memory", "uma", "numa", "shared memory architecture"],
                "examples": ["Multi-core CPUs", "SMP systems", "NUMA servers"]
            },
            "distributed memory": {
                "answer": "**Distributed Memory Architecture** has each processor with its own private memory.\n\n**Characteristics:**\n- Processors communicate via message passing\n- No global memory view\n- Better scalability than shared memory\n\n**Communication:** Point-to-point, Collective operations, One-sided",
                "keywords": ["distributed memory", "message passing", "clusters", "distributed memory architecture"],
                "examples": ["Computer clusters", "Grid computing", "Supercomputers"]
            },
            "flynn's taxonomy": {
                "answer": "**Flynn's Taxonomy** classifies computer architectures:\n\n1. **SISD** - Single Instruction, Single Data (Traditional CPUs)\n2. **SIMD** - Single Instruction, Multiple Data (GPUs, Vector processors)\n3. **MISD** - Multiple Instruction, Single Data (Rare, fault-tolerant)\n4. **MIMD** - Multiple Instruction, Multiple Data (Multi-core, clusters)\n\n**Most common:** SIMD (GPUs) and MIMD (multi-core CPUs)",
                "keywords": ["flynn", "flynn's taxonomy", "sisd", "simd", "misd", "mimd", "taxonomy"],
                "examples": ["SISD: Traditional uniprocessors", "SIMD: GPU architectures", "MIMD: Multi-core processors"]
            }
        }
        
        print("✓ Knowledge base loaded with", len(self.knowledge_base), "topics")
        print("=" * 50)
    
    def search_web(self, query):
        """Search the web using Tavily API"""
        if not self.tavily_api_key:
            return "## 🔍 Web Search (Disabled)\n\nWeb search requires a Tavily API key. Add 'TAVILY_API_KEY=your_key' to .env file."
        
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 3,
                "include_images": False,
                "include_answer": True
            }
            headers = {"Content-Type": "application/json"}
            
            print(f"🔍 Searching web for: {query}")
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            web_info = "## 🌐 Web Search Results\n\n"
            
            if data.get('answer'):
                web_info += f"**Summary:** {data['answer']}\n\n"
            
            if data.get('results'):
                web_info += "**Sources:**\n"
                for i, result in enumerate(data['results'][:3], 1):
                    title = result.get('title', 'Untitled')
                    url = result.get('url', 'No URL')
                    content = result.get('content', 'No content available')[:200]
                    web_info += f"{i}. **{title}**\n   {url}\n   {content}...\n\n"
            else:
                web_info += "No relevant web results found.\n"
            
            web_info += f"\n*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
            return web_info
            
        except requests.exceptions.Timeout:
            return "⚠ Web search timed out. Try again."
        except requests.exceptions.RequestException as e:
            return f"⚠ Web search error: {str(e)[:100]}"
        except Exception as e:
            return f"⚠ Error: {str(e)[:100]}"
    
    def query(self, question, use_web_search=False):
        """Process a user question - FIXED VERSION"""
        try:
            question_lower = question.lower().strip()
            
            print(f"🔍 Question received: '{question}'")
            print(f"🔍 Question lower: '{question_lower}'")
            
            # Initialize variables
            answer = None
            matched_topic = None
            sources = []
            web_info = ""
            
            # FIRST: Check for exact matches in knowledge base
            for topic_name, topic_info in self.knowledge_base.items():
                # Check if topic name is in question
                if topic_name in question_lower:
                    print(f"✅ Exact match found: {topic_name}")
                    answer = topic_info['answer']
                    matched_topic = topic_name
                    sources.append({
                        "source": "Course Materials",
                        "content": topic_name.capitalize()
                    })
                    break
            
            # SECOND: If no exact match, check keywords
            if not answer:
                print("🔍 No exact match, checking keywords...")
                best_match = None
                best_score = 0
                
                for topic_name, topic_info in self.knowledge_base.items():
                    # Check all keywords for this topic
                    for keyword in topic_info.get('keywords', [topic_name]):
                        if keyword in question_lower:
                            # Score based on keyword length (longer keywords are more specific)
                            score = len(keyword)
                            if score > best_score:
                                best_score = score
                                best_match = topic_name
                                print(f"✅ Keyword match: {keyword} -> {topic_name} (score: {score})")
                
                if best_match:
                    print(f"✅ Best match selected: {best_match}")
                    topic_info = self.knowledge_base[best_match]
                    answer = topic_info['answer']
                    matched_topic = best_match
                    sources.append({
                        "source": "Course Materials",
                        "content": best_match.capitalize()
                    })
            
            # THIRD: If still no match, check for partial matches
            if not answer:
                print("🔍 No keyword match, checking partial matches...")
                words_in_question = set(question_lower.split())
                
                for topic_name, topic_info in self.knowledge_base.items():
                    # Check each word in question against topic
                    for word in words_in_question:
                        if len(word) > 4:  # Only check meaningful words
                            if word in topic_name or any(word in kw for kw in topic_info.get('keywords', [])):
                                print(f"✅ Partial match: '{word}' in '{topic_name}'")
                                answer = topic_info['answer']
                                matched_topic = topic_name
                                sources.append({
                                    "source": "Course Materials",
                                    "content": topic_name.capitalize()
                                })
                                break
                    if answer:
                        break
            
            # FOURTH: If still no answer, use default welcome message
            if not answer:
                print("❌ No match found, using default response")
                answer = """🤖 **Welcome to Parallel Computing Assistant!**

I can help you understand:
• **Parallel Architectures** - Shared vs Distributed memory
• **Programming Models** - MPI, OpenMP, CUDA, MapReduce
• **Key Concepts** - Amdahl's Law, Load balancing, Synchronization
• **Applications** - Scientific computing, Machine learning, Big data

**Try asking specific questions like:**
• "What is MPI?"
• "Explain CUDA programming"
• "What is Amdahl's Law?"
• "How does OpenMP work?"
• "What is distributed computing?"

*Ask me anything about parallel and distributed computing!*"""
            
            # Add examples if we have a matched topic
            if matched_topic and 'examples' in self.knowledge_base[matched_topic]:
                answer += f"\n\n**Examples of {matched_topic.capitalize()}:**\n"
                for example in self.knowledge_base[matched_topic]['examples']:
                    answer += f"• {example}\n"
            
            # FIFTH: Add web search if requested
            if use_web_search:
                print("🔍 Web search requested...")
                web_info = self.search_web(question)
                if web_info and "Web search requires" not in web_info:
                    answer += f"\n\n{'='*50}\n\n{web_info}"
                    sources.append({
                        "source": "Web Search",
                        "content": "Current information from web"
                    })
                elif "Web search requires" in web_info:
                    answer += f"\n\n{web_info}"
            
            print(f"✅ Answer prepared. Topic: {matched_topic or 'Default'}")
            
            return {
                "answer": answer,
                "sources": sources,
                "metadata": {
                    "used_web_search": use_web_search and bool(web_info),
                    "matched_topic": matched_topic,
                    "has_tavily_key": bool(self.tavily_api_key),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"❌ Error in query: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "answer": f"⚠ Error: {str(e)}",
                "sources": [],
                "metadata": {
                    "used_web_search": False,
                    "timestamp": datetime.now().isoformat()
                }
            }

# ========== CREATE CHATBOT INSTANCE ==========
chatbot_instance = RAGChatbot()

# ========== FLASK APP ==========
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        "message": "🎓 Parallel Computing Chatbot API",
        "status": "running",
        "version": "2.0",
        "features": [
            "Course materials knowledge base",
            "Tavily web search integration",
            "Improved keyword matching"
        ],
        "endpoints": {
            "GET /api/health": "Check API status",
            "POST /api/chat": "Ask questions (include web_search: true/false)",
            "GET /api/sample-questions": "Get example questions"
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "parallel-computing-chatbot",
        "timestamp": datetime.now().isoformat(),
        "knowledge_base": {
            "topics": len(chatbot_instance.knowledge_base),
            "web_search_available": bool(os.getenv("TAVILY_API_KEY"))
        }
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "Invalid JSON",
                "answer": "Please send valid JSON data."
            }), 400
        
        question = data.get('question', '').strip()
        web_search = data.get('web_search', False)
        
        if not question:
            return jsonify({
                "error": "Empty question",
                "answer": "Please ask a question about parallel computing."
            }), 400
        
        print(f"📩 Question: '{question}' | Web search: {web_search}")
        
        # Get response from chatbot
        response = chatbot_instance.query(question, use_web_search=web_search)
        
        return jsonify({
            "answer": response["answer"],
            "sources": response["sources"],
            "metadata": response["metadata"],
            "question": question
        })
        
    except Exception as e:
        print(f"❌ Error in /api/chat: {e}")
        return jsonify({
            "error": str(e),
            "answer": "Sorry, I encountered an error. Please try again."
        }), 500

@app.route('/api/sample-questions', methods=['GET'])
def sample_questions():
    questions = {
        "basic_concepts": [
            "What is parallel computing?",
            "Explain distributed computing",
            "What is the difference between MPI and OpenMP?",
            "How does CUDA work?",
            "Explain Amdahl's Law with example",
            "What is load balancing in parallel systems?",
            "Explain shared memory vs distributed memory",
            "What is Flynn's taxonomy?"
        ],
        "web_search_topics": [
            "Recent developments in GPU computing",
            "Latest trends in distributed machine learning",
            "Current state of quantum computing for parallelism",
            "New parallel programming frameworks 2024"
        ],
        "practical_applications": [
            "How is parallel computing used in machine learning?",
            "Parallel computing examples in scientific research",
            "Big data processing with MapReduce",
            "Cloud computing and parallelism"
        ]
    }
    
    return jsonify(questions)

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Get all available topics in knowledge base"""
    topics = list(chatbot_instance.knowledge_base.keys())
    return jsonify({
        "topics": topics,
        "count": len(topics),
        "description": "Available topics in parallel computing knowledge base"
    })

@app.route('/api/debug-match', methods=['POST'])
def debug_match():
    """Debug endpoint to see how questions are matched"""
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    question_lower = question.lower()
    matches = []
    
    for topic_name, topic_info in chatbot_instance.knowledge_base.items():
        # Check exact match
        if topic_name in question_lower:
            matches.append({
                "topic": topic_name,
                "type": "exact_match",
                "score": 100
            })
        # Check keyword matches
        for keyword in topic_info.get('keywords', [topic_name]):
            if keyword in question_lower:
                matches.append({
                    "topic": topic_name,
                    "type": "keyword_match",
                    "keyword": keyword,
                    "score": len(keyword) * 10
                })
    
    return jsonify({
        "question": question,
        "question_lower": question_lower,
        "matches": matches,
        "total_matches": len(matches)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n Starting server on http://localhost:{port}")
    print(" Press Ctrl+C to stop")
    print(f" Knowledge base: {len(chatbot_instance.knowledge_base)} topics")
    print("✅ Ready to answer questions!\n")
    app.run(host='0.0.0.0', port=port, debug=True)