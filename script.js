// ==================== CONFIGURATION ====================
const API_BASE_URL = 'http://localhost:5000/api';
console.log(' Frontend loaded. API URL:', API_BASE_URL);

// ==================== STATE VARIABLES ====================
let isWebSearchEnabled = false;
let isProcessing = false;

// ==================== DOM ELEMENTS ====================
const elements = {
    chatMessages: document.getElementById('chatMessages'),
    userInput: document.getElementById('userInput'),
    sendButton: document.getElementById('sendButton'),
    statusText: document.getElementById('statusText'),
    statusDot: document.querySelector('.status-dot'),
    webSearchToggle: document.getElementById('webSearchToggle'),
    searchIndicator: document.getElementById('searchIndicator')
};

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', function() {
    console.log('📱 DOM Content Loaded');
    
    // Initialize the chat
    initializeChat();
    
    // Setup event listeners
    setupEventListeners();
    
    // Check backend connection
    checkBackendConnection();
    
    // Add welcome message
    addWelcomeMessage();
});

// ==================== INITIALIZATION FUNCTIONS ====================
function initializeChat() {
    console.log(' Initializing chat...');
    
    // Clear any existing messages (except welcome)
    if (elements.chatMessages) {
        // Keep only welcome message if exists
        const welcomeMsg = elements.chatMessages.querySelector('.welcome-message');
        elements.chatMessages.innerHTML = '';
        if (welcomeMsg) {
            elements.chatMessages.appendChild(welcomeMsg);
        }
    }
}

function setupEventListeners() {
    console.log('🎮 Setting up event listeners...');
    
    // Send button click
    if (elements.sendButton) {
        elements.sendButton.addEventListener('click', sendMessage);
        console.log('✓ Send button listener added');
    }
    
    // Enter key in textarea
    if (elements.userInput) {
        elements.userInput.addEventListener('keydown', handleKeyDown);
        elements.userInput.addEventListener('input', autoResizeTextarea);
        console.log('✓ Textarea listeners added');
    }
    
    // Web search toggle
    if (elements.webSearchToggle) {
        elements.webSearchToggle.addEventListener('change', toggleWebSearch);
        console.log('✓ Web search toggle listener added');
    }
    
    // Focus on input
    setTimeout(() => {
        if (elements.userInput) {
            elements.userInput.focus();
        }
    }, 500);
}

// ==================== CHAT FUNCTIONS ====================
async function sendMessage() {
    if (isProcessing) {
        console.log(' Already processing, skipping...');
        return;
    }
    
    const message = elements.userInput.value.trim();
    if (!message) {
        console.log('⚠ Empty message, skipping...');
        return;
    }
    
    console.log(' Sending message:', message.substring(0, 50) + '...');
    
    // Add user message
    addMessage(message, 'user');
    
    // Clear input
    elements.userInput.value = '';
    autoResizeTextarea();
    
    // Show typing indicator
    showTypingIndicator();
    
    // Set processing flag
    isProcessing = true;
    if (elements.sendButton) elements.sendButton.disabled = true;
    
    try {
        // Prepare request data
        const requestData = {
            question: message,
            web_search: isWebSearchEnabled
        };
        
        console.log(' Request data:', requestData);
        
        // Send request to backend
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        console.log(' Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log(' Response received:', data ? 'Yes' : 'No');
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Add bot response
        if (data.answer) {
            addMessage(data.answer, 'bot');
        } else {
            addMessage('Sorry, I could not generate a response. Please try again.', 'bot');
        }
        
    } catch (error) {
        console.error(' Error sending message:', error);
        removeTypingIndicator();
        addMessage(`Error: ${error.message}. Please check if the backend server is running.`, 'bot');
    } finally {
        // Reset processing flag
        isProcessing = false;
        if (elements.sendButton) elements.sendButton.disabled = false;
        
        // Focus back on input
        if (elements.userInput) {
            elements.userInput.focus();
        }
    }
}

// ==================== MESSAGE FUNCTIONS ====================
function addMessage(text, sender) {
    if (!elements.chatMessages) {
        console.error(' chatMessages element not found!');
        return;
    }
    
    console.log(` Adding ${sender} message`);
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    // Create message content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Format the text (preserve line breaks)
    const formattedText = text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    contentDiv.innerHTML = formattedText;
    
    // Create timestamp
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = getCurrentTime();
    
    // Assemble message
    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);
    
    // Add to chat
    elements.chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    scrollToBottom();
}

function addWelcomeMessage() {
    const welcomeMessage = `🤖 **Welcome to Parallel Computing Assistant!**

I can help you understand:
• **Parallel Architectures** - Shared vs Distributed memory
• **Programming Models** - MPI, OpenMP, CUDA, MapReduce
• **Key Concepts** - Amdahl's Law, Load balancing, Synchronization
• **Applications** - Scientific computing, Machine learning, Big data

**Try asking:**
• "What is the difference between MPI and OpenMP?"
• "How does CUDA work?"
• "Explain Amdahl's Law"
• "What are recent trends in parallel computing?"`;
    
    addMessage(welcomeMessage, 'bot');
}

function showTypingIndicator() {
    if (!elements.chatMessages) return;
    
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typingIndicator';
    typingDiv.className = 'typing-indicator';
    
    typingDiv.innerHTML = `
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
        <span>Assistant is typing${isWebSearchEnabled ? ' and searching web...' : '...'}</span>
    `;
    
    elements.chatMessages.appendChild(typingDiv);
    scrollToBottom();
}

function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator && typingIndicator.parentNode) {
        typingIndicator.parentNode.removeChild(typingIndicator);
    }
}

// ==================== HELPER FUNCTIONS ====================
function handleKeyDown(event) {
    // Enter to send, Shift+Enter for new line
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResizeTextarea() {
    if (!elements.userInput) return;
    
    elements.userInput.style.height = 'auto';
    const newHeight = Math.min(elements.userInput.scrollHeight, 120);
    elements.userInput.style.height = newHeight + 'px';
}

function toggleWebSearch() {
    isWebSearchEnabled = elements.webSearchToggle.checked;
    
    if (elements.searchIndicator) {
        elements.searchIndicator.style.display = isWebSearchEnabled ? 'inline-flex' : 'none';
    }
    
    console.log(`🌐 Web search ${isWebSearchEnabled ? 'enabled' : 'disabled'}`);
    
    // Show notification
    const notification = isWebSearchEnabled 
        ? 'Web search enabled. I will search for current information.'
        : 'Web search disabled. Using course materials only.';
    
    addMessage(notification, 'bot');
}

function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function scrollToBottom() {
    if (elements.chatMessages) {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }
}

// ==================== CONNECTION FUNCTIONS ====================
async function checkBackendConnection() {
    console.log('🔌 Checking backend connection...');
    
    if (elements.statusText) {
        elements.statusText.textContent = 'Connecting...';
    }
    if (elements.statusDot) {
        elements.statusDot.className = 'status-dot';
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Backend connected:', data);
            
            if (elements.statusText) {
                elements.statusText.textContent = 'Connected';
            }
            if (elements.statusDot) {
                elements.statusDot.classList.add('connected');
            }
            
            return true;
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
        
    } catch (error) {
        console.error('❌ Backend connection failed:', error);
        
        if (elements.statusText) {
            elements.statusText.textContent = 'Disconnected';
        }
        if (elements.statusDot) {
            elements.statusDot.classList.add('disconnected');
        }
        
        // Show error message
        addMessage('⚠️ Backend server is not connected. Please make sure the backend is running on port 5000.', 'bot');
        
        return false;
    }
}

// ==================== QUICK ACTION FUNCTIONS ====================
function askQuestion(question) {
    if (!elements.userInput) return;
    
    elements.userInput.value = question;
    autoResizeTextarea();
    elements.userInput.focus();
    
    // Auto-send after a short delay
    setTimeout(() => {
        sendMessage();
    }, 100);
}

function askTopic(topic) {
    const question = `Explain ${topic} in parallel computing`;
    askQuestion(question);
}

// ==================== GLOBAL FUNCTIONS (for HTML onclick) ====================
window.askQuestion = askQuestion;
window.askTopic = askTopic;