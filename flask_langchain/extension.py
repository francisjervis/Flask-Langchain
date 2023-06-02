from flask import g, current_app, session
from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    _message_to_dict,
    messages_from_dict,
)
from langchain.memory.chat_memory import BaseChatMemory, BaseMemory
from langchain.memory.utils import get_prompt_input_key
from langchain.embeddings import OpenAIEmbeddings
from sqlalchemy import Column
import datetime
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
from pydantic import Field
from flask_sqlalchemy import SQLAlchemy
from langchain.vectorstores import Chroma


db = SQLAlchemy()
app = current_app

class FlaskChatMessage(db.Model):
    __tablename__ = 'message_store'
    id = Column(db.Integer, primary_key=True)
    langchain_session_id = Column(db.Text)
    user_id = Column(db.Text, nullable=True)
    conversation_id = Column(db.Text, nullable=True)
    message = Column(db.Text)
    message_kwargs = Column(db.Text)
    type = Column(db.Text)
    timestamp = Column(db.DateTime)

class FlaskChatMessageHistory(BaseChatMessageHistory):
    """
    A class that represents a chat message history using Flask-SQLAlchemy.
    Inherits from BaseChatMessageHistory.
    """

    def __init__(
            self,
            db,
            table_name: str = "message_store",
            langchain_session_id: str = None,
            langchain_user_id: str = None,
            langchain_conversation_id: str = None,
    ):
        """
        Initialize the FlaskChatMessageHistory object with the given Flask-SQLAlchemy db.

        :param db: A Flask-SQLAlchemy db object.
        :param table_name: The name of the table to store chat messages. Defaults to "message_store".
        :param langchain_session_id: The session ID to use for this chat message history.
        :param user_id: The user ID to use for this chat message history.
        :param conversation_id: The conversation ID to use for this chat message history.

        """

        self.table_name = table_name
        self.db = db
        self.langchain_session_id = langchain_session_id
        self.langchain_user_id = langchain_user_id
        self.langchain_conversation_id = langchain_conversation_id

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all messages from db"""
        print("retrieving messages")

        if self.langchain_session_id is None:
            self.langchain_session_id = session.get('langchain_session_id', None)
        if self.langchain_user_id is None:
            self.langchain_user_id = session.get('langchain_user_id', None)
        if self.langchain_conversation_id is None:
            self.langchain_conversation_id = session.get('langchain_conversation_id', None)

        print("getting messages for conversation id: ", self.langchain_conversation_id)

        found_messages = self.db.query(FlaskChatMessage).filter(FlaskChatMessage.conversation_id == self.langchain_conversation_id).all()

        if found_messages is None or len(found_messages) == 0:
            print("no messages found")
            return []

        print("found messages: ", found_messages)
        for message in found_messages:
            print("found message content: ", message.message)

        messages = []
        for message in found_messages:
            print("found message content: ", message.message)
            if message.type == "human":
                messages.append(HumanMessage(content=message.message, additional_kwargs={}))
            elif message.type == "ai":
                messages.append(AIMessage(content=message.message, additional_kwargs={}))
            # elif message.type == "system":
            #     messages.append(SystemMessage(content=message.message))
            # elif message.type == "chat":
            #     messages.append(ChatMessage(content=message.message))
            else:
                raise ValueError("Unknown message type: " + message.type)

        return messages

    def add_user_message(self, message: str, langchain_session_id: str = None, langchain_user_id: str = None,
                         langchain_conversation_id: str = None) -> None:
        with app.app_context():
            if langchain_session_id is None:
                langchain_session_id = session.get('langchain_session_id', None)
            if langchain_user_id is None:
                langchain_user_id = session.get('langchain_user_id', None)
            if langchain_conversation_id is None:
                langchain_conversation_id = session.get('langchain_conversation_id', None)

        self.append(HumanMessage(content=message), langchain_session_id=langchain_session_id, langchain_user_id=langchain_user_id,
                    langchain_conversation_id=langchain_conversation_id)

    def add_ai_message(self, message: str, langchain_session_id: str = None, user_id: str = None,
                       conversation_id: str = None) -> None:
        self.append(AIMessage(content=message), langchain_session_id=langchain_session_id, langchain_user_id=user_id,
                    langchain_conversation_id=conversation_id)

    def append(self, message: FlaskChatMessage, langchain_session_id: str, langchain_user_id: str = None,
               langchain_conversation_id: str = None) -> None:
        """Append the message to the record in db"""
        with app.app_context():
            if langchain_session_id is None:
                langchain_session_id = session.get('langchain_session_id', None)
            if langchain_user_id is None:
                langchain_user_id = session.get('langchain_user_id', None)
            if langchain_conversation_id is None:
                langchain_conversation_id = session.get('langchain_conversation_id', None)

        with self.db as dbsession:
            m = _message_to_dict(message)
            print("message to dict: ", m)
            print(type(m))
            content = m['data']['content']
            additional_kwargs = m['data']['additional_kwargs']
            example = m['data']['example']
            m_type = m['type']

            print("adding message to db: " + content, langchain_session_id, langchain_user_id, langchain_conversation_id)
            dbsession.add(
                FlaskChatMessage(
                    langchain_session_id=langchain_session_id,
                    user_id=langchain_user_id,
                    conversation_id=langchain_conversation_id,
                    message=content,
                    message_kwargs=additional_kwargs,
                    type=m_type,
                    timestamp=datetime.datetime.now(),
                ))
            dbsession.commit()

    def clear(self) -> None:
        """Clear session memory from db"""

        with self.Session() as session:
            session.query(self.Message).filter(
                self.Message.langchain_session_id == self.langchain_session_id
            ).delete()
            session.commit()

class LangchainFlaskMemory:
    """
    A Flask extension that provides a decorator to make a FlaskChatMessageHistory object
    initialized with the current app's Flask-SQLAlchemy db available to the decorated method, and make a
    FlaskChromaVectorstore object available to the decorated method.
    """

    def __init__(self, app=None, db=None):
        """
        Initialize the ChatMemory extension with the given Flask app and Flask-SQLAlchemy db.

        :param app: A Flask app object. Defaults to None.
        :param db: A Flask-SQLAlchemy db object. Defaults to None.
        """
        self.app = app
        self.db = db
        self.message_store_table = None
        if app is not None and db is not None:
            self.init_app(app, db)

    def init_app(self, app, db):
        """
        Initialize the ChatMemory extension with the given Flask app and Flask-SQLAlchemy db.

        :param app: A Flask app object.
        :param db: A Flask-SQLAlchemy db object.
        """
        self.app = app
        self.db = db

        # if the extension has already been initialized, use that
        # otherwise, initialize it
        with self.app.app_context():
            if hasattr(app.extensions, "langchain_chat_memory"):
                print("found existing langchain_chat_memory")
                return app.extensions["langchain_chat_memory"]

        with self.app.app_context():
            self.dbSession = db.session()
            self.message_store_table = self._create_table_if_not_exists()
            app.extensions["langchain_chat_memory"] = self
    #         register before_request and teardown_request handlers
            app.before_request(self.before_request)

    def before_request(self):
        """
        Get or create session and conversation ids for the current request.
        Make the session permanent.
        :return:
        """
        langchain_session_id = session.get('langchain_session_id', None)
        if langchain_session_id is None:
            g.langchain_session_id = str(uuid.uuid4())
            session['langchain_session_id'] = g.langchain_session_id

        langchain_conversation_id = session.get('langchain_conversation_id', None)

        if langchain_conversation_id is None:
            langchain_conversation_id = self._create_new_conversation_id()
            session['langchain_conversation_id'] = langchain_conversation_id

        session.modified = True
        session.permanent = True

    @property
    def chat_memory(self) -> FlaskChatMessageHistory:
        """
        Return a FlaskChatMessageHistory object for the current request.
        :param langchain_session_id:
        :param user_id:
        :param conversation_id:
        :return: langchain_chat_memory
        """
        langchain_session_id = session.get('langchain_session_id', None)
        user_id = session.get('langchain_user_id', None)
        conversation_id = session.get('langchain_conversation_id', None)

        return self._create_chat_memory(langchain_session_id=langchain_session_id, user_id=user_id, conversation_id=conversation_id)

    def _create_chat_memory(self, langchain_session_id: str = None, user_id: str = None,
                            conversation_id: str = None) -> FlaskChatMessageHistory:
        """
        Create a new ChatMessageHistory object with the current app's Flask-SQLAlchemy db.

        :return: An instance of the FlaskChatMessageHistory object.
        """
        return FlaskChatMessageHistory(self.db.session(), langchain_session_id=langchain_session_id, langchain_user_id=user_id, langchain_conversation_id=conversation_id)

    def _create_table_if_not_exists(self):
        """
        Create the chat message table if it does not exist.

        :return: An instance of the MessageStore table object.
        """
        with current_app.app_context():
            self.db.create_all()

    def _create_new_conversation_id(self):
        """
        Create a new conversation id and add it to the session.
        """
        if session.get('langchain_user_id', None) is None:
            raise Exception("langchain_user_id is not set")
        session['langchain_conversation_id'] = str(uuid.uuid4())
        return session['langchain_conversation_id']

    def set_user_id(self, user_id:str):
        """
        Set the user id for the current request.
        """
        session['langchain_user_id'] = user_id
        self._create_new_conversation_id()

    def log_out_user(self):
        """
        Clear the user id for the current request.
        """
        session.pop('langchain_user_id', None)
        session.pop('langchain_conversation_id', None)

    @property
    def chroma_vector_store(self) -> Chroma:
        """
        Return a langchain chroma client object for the current request.
        """
        return self._create_chroma_vector_store()

    def _create_chroma_vector_store(self, user_id:str = None, db_impl:str = "duckdb+parquet", persist_dir:str = "./langchain_chromadb",
                                    ) -> Chroma:
        """
        Create a new langchain chroma client object with the current user id.
        :return: An instance of the langchain chroma client object.
        """

        if user_id is None:
            user_id = session.get('langchain_user_id', None)

        if user_id is None:
            raise Exception("user_id is not set")

        openai_api_key = os.getenv("OPENAI_API_KEY", None)

        if openai_api_key is None:
            chroma = Chroma(persist_directory=persist_dir, collection_name=user_id)

        else:
            embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                model="text-embedding-ada-002",
            )

            chroma = Chroma(persist_directory=persist_dir, collection_name=user_id, embedding_function=embeddings)

        return chroma
class ConversationFlaskMemory(BaseChatMemory):
    """
    Langchain memory for storing conversation state.
    """

    chat_memory: FlaskChatMessageHistory = Field(default_factory=FlaskChatMessageHistory)
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    return_messages: bool = False

    memory_key: str = "history"  #: :meta private:
    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def _get_input_output(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        print("ConversationFlaskMemory _get_input_output")
        print(inputs)
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Return history buffer."""
        return {self.memory_key: self.chat_memory.messages}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()
