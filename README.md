# pdc-chatbot

# project structure
pdc-ccp/ (or parallel-computing-chatbot/)
├── backend/
│   ├── app.py
│   ├── rag_system.py
│   ├── knowledge_base/
│   ├── requirements.txt
│   └── .env # place your api keys here
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── serve.py  

# first run backend on vs studio terminal
python app.py

# then frontend on vs studio terminal
python -m http.server 8000
