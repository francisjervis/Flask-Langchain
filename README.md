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

# Overview
Flask-Langchain adds a session and conversation id to the Flask session object, along with a user id if provided.
After the extension is initialized, the LangchainFlaskMemory object exposes chat_memory and chroma_vector_store properties
which can be used to create ConversationFlaskMemory and ChromaVectorStore objects, respectively.
Messages are retrieved by conversation id, and document collections are organized by user id. If no user id is provided,
the session id is used instead, allowing for anonymous users (note that some LLM providers, including OpenAI, discourage 
the use of their APIs in unauthenticated contexts).

These classes inherit from the Langchain BaseChatMemory and ChromaVectorStore classes, and can be used in the same way.

# Installation

Coming soon - for now, clone the repo and import the Flask-Langchain.extension module.

Requires Flask-SQLAlchemy.

# Known issues

- Incorrect formatting of conversation history (as a string representation of a list of BaseMessages).
- After it is initialized, you have to access the LangchainFlaskMemory object using `current_app.extensions['langchain_chat_memory']` - this could be improved
- Test coverage may be incomplete
- Supports in-memory Chroma database only

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

        app.run(debug=True)
```


