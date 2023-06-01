import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from langchain.schema import (
    AIMessage,
    HumanMessage,
    BaseChatMessageHistory,
    BaseMessage,
)
from langchain.memory.chat_memory import BaseMemory
from langchain.memory.utils import get_prompt_input_key


from flask_langchain.extension import *


@pytest.fixture(scope="module")
def app():
    app = Flask("test_app")
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////tmp/test.db"
    return app


@pytest.fixture(scope="module")
def db(app):
    return SQLAlchemy(app)


def test_FlaskChatMessageHistory_init(app, db):
    message_history = FlaskChatMessageHistory(
        db,
        table_name="message_store",
        langchain_session_id="session_id",
        langchain_user_id="user_id",
        langchain_conversation_id="conversation_id",
    )

    assert message_history.table_name == "message_store"
    assert message_history.db == db
    assert message_history.langchain_session_id == "session_id"
    assert message_history.langchain_user_id == "user_id"
    assert message_history.langchain_conversation_id == "conversation_id"


@patch("langchain_chatservice.FlaskChatMessageHistory.messages", return_value=[])
def test_FlaskChatMessageHistory_messages(mock_messages, app, db):
    message_history = FlaskChatMessageHistory(db)
    messages = message_history.messages

    assert messages == []
    mock_messages.assert_called_once()


@patch("langchain_chatservice.FlaskChatMessageHistory.append")
def test_FlaskChatMessageHistory_add_user_message(mock_append, app, db):
    message_history = FlaskChatMessageHistory(db)
    message = HumanMessage(content="hello")
    message_history.add_user_message("hello", langchain_session_id="session_id", langchain_user_id="user_id",
                                     langchain_conversation_id="conversation_id")

    mock_append.assert_called_once_with(message, langchain_session_id="session_id", langchain_user_id="user_id",
                                        langchain_conversation_id="conversation_id")


@patch("langchain_chatservice.FlaskChatMessageHistory.append")
def test_FlaskChatMessageHistory_add_ai_message(mock_append, app, db):
    message_history = FlaskChatMessageHistory(db)
    message = AIMessage(content="hello, I'm AI")
    message_history.add_ai_message("hello, I'm AI", langchain_session_id="session_id", user_id="user_id",
                                   conversation_id="conversation_id")

    mock_append.assert_called_once_with(message, langchain_session_id="session_id", langchain_user_id="user_id",
                                        langchain_conversation_id="conversation_id")


@patch("langchain_chatservice.json.dumps", return_value="{'content': 'hello'}")
@patch("langchain_chatservice.db.session.add")
@patch("langchain_chatservice.db.session.commit")
def test_FlaskChatMessageHistory_append(mock_commit, mock_add, mock_dumps, app, db):
    message_history = FlaskChatMessageHistory(db)
    message = HumanMessage(content="hello")
    message_history.append(message, langchain_session_id="session_id", langchain_user_id="user_id",
                           langchain_conversation_id="conversation_id")

    mock_dumps.assert_called_once()
    mock_add.assert_called_once()
    mock_commit.assert_called_once()


@patch("langchain_chatservice.db.session.query")
@patch("langchain_chatservice.db.session.commit")
def test_FlaskChatMessageHistory_clear(mock_commit, mock_query, app, db):
    message_history = FlaskChatMessageHistory(db)
    message_history.clear()

    mock_query.assert_called_once()
    mock_commit.assert_called_once()


def test_LangchainFlaskMemory_init(app, db):
    memory = LangchainFlaskMemory(app=app, db=db)

    assert memory.app == app
    assert memory.db == db


def test_LangchainFlaskMemory_chat_memory(app, db):
    memory = LangchainFlaskMemory(app=app, db=db)

    chat_memory = memory.chat_memory
    assert isinstance(chat_memory, FlaskChatMessageHistory)


def test_ConversationFlaskMemory_load_memory_variables():
    conv_memory = ConversationFlaskMemory()
    conv_memory.chat_memory = Mock()

    result = conv_memory.load_memory_variables({"test_key": "test_value"})

    assert result == {conv_memory.memory_key: conv_memory.chat_memory.messages}


@patch("langchain_chatservice.ConversationFlaskMemory._get_input_output", return_value=("input", "output"))
def test_ConversationFlaskMemory_save_context(mock_get_input_output):
    conv_memory = ConversationFlaskMemory()
    conv_memory.chat_memory = Mock()
    inputs = {"test_key": "test_value"}
    outputs = {"test_output_key": "test_output_value"}

    conv_memory.save_context(inputs, outputs)

    mock_get_input_output.assert_called_once_with(inputs, outputs)
    conv_memory.chat_memory.add_user_message.assert_called_once_with("input")
    conv_memory.chat_memory.add_ai_message.assert_called_once_with("output")


def test_ConversationFlaskMemory_clear():
    conv_memory = ConversationFlaskMemory()
    conv_memory.chat_memory = Mock()

    conv_memory.clear()

    conv_memory.chat_memory.clear.assert_called_once()
