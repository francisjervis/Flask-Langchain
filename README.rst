Flask-Langchain
===============

Example usage
-------------

.. code-block:: python
   :linenos:

   from flask import Flask, request, jsonify
   from flask_sqlalchemy import SQLAlchemy
   import os
   from langchainmemory import LangchainFlaskMemory, ConversationFlaskMemory
   from langchain.llms import OpenAI
   from langchain.chains import ConversationChain

   app = Flask(__name__)

   app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['SQLALCHEMY_DATABASE_URI']
   app.secret_key = "wwiegnwiegwinEGInwelgnweG"
   db = SQLAlchemy(app)

   langchain_memory = LangchainFlaskMemory(app, db)

   llm = OpenAI(temperature=0)

   @app.route('/')
   def index():
       langchain_memory.set_user_id("abc123")
       return "Hello, World!"

   @app.route('/chat', methods=['POST'])
   def chat():
       input = request.json['message']
       m = langchain_memory._create_chat_memory()
       c = ConversationFlaskMemory(chat_memory=m, return_messages=True)
       conversation = ConversationChain(
           llm=llm,
           verbose=True,
           memory=c,
       )
       answer = conversation.predict(input=input)
       return jsonify({"message": answer})

   @app.route('/add')
   def add():
       chroma = langchain_memory._create_chroma_vector_store()
       chroma.add_texts(texts=["doc1", "doc2", "doc3"])
       return "Added texts"

   @app.route('/count')
   def count():
       chroma = langchain_memory._create_chroma_vector_store()
       return str(chroma._collection.count())
