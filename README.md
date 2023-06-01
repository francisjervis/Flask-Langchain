# Flask-Langchain

**Pre-release version. Not ready for production use.**

Flask-Langchain is a Flask extension that provides a simple interface for
using [Langchain](https://github.com/hwchase17/langchain) with Flask.
Currently, it provides an SQLAlchemy based memory class for storing conversation histories,
on a per-user or per-session basis, and a ChromaVectorStore class for storing document vectors (per-user only).

To use, simply create a LangchainFlaskMemory object, passing in your Flask app and SQLAlchemy db object.
Then, use the LangchainFlaskMemory object to create a ConversationFlaskMemory object, which can be passed
to a Langchain chain as the memory parameter.

Set the user id using the set_user_id method in your app's login callback. If no user id is set, the session id will be used.

# Installation

Coming soon - for now, clone the repo and import the langchainmemory module.

# Example usage

```
    from flask import Flask, request, jsonify
    from flask_sqlalchemy import SQLAlchemy
    import os
    from langchainmemory import LangchainFlaskMemory, ConversationFlaskMemory
    from langchain.llms import OpenAI
    from langchain.chains import ConversationChain

    app = Flask(__name__)

    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['SQLALCHEMY_DATABASE_URI']
    app.secret_key = "supersecretstring"
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
       
    if __name__ == '__main__':

        app.run()
```


