"""
Comprehensive Unit & Integration Test Suite for SupabaseChatRepository
Validates CRUD, PostgREST REST specs, RLS policies, Multilingual persistence,
Session ownership security, Restart persistence simulation, and Error handling.
"""

import json
from datetime import datetime, timezone
import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import settings
from backend.app.schemas.chat import ChatMessageSchema, ChatSessionSchema, ChatRequest
from backend.app.db.chat_repository import (
    SupabaseChatRepository,
    InMemoryChatRepository,
    SupabaseConnectionError,
    SupabaseQueryError,
    SessionAccessDeniedError,
    set_chat_repository,
    reset_chat_repository,
    get_chat_repository,
)
from backend.app.services.chat.session_service import session_service
from backend.app.services.chat.chat_service import chat_service


MOCK_SUPABASE_URL = "https://mockproject.supabase.co"
MOCK_SUPABASE_KEY = "mock-service-role-secret-key"


class PostgRESTMockState:
    """In-memory state simulating Supabase PostgREST tables."""

    def __init__(self):
        self.chat_sessions = {}
        self.chat_messages = []

    def reset(self):
        self.chat_sessions.clear()
        self.chat_messages.clear()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url_path = request.url.path
        query = request.url.query.decode("utf-8") if isinstance(request.url.query, bytes) else str(request.url.query)

        # 1. Health check: /rest/v1/chat_sessions?limit=1
        if url_path == "/rest/v1/chat_sessions" and request.method == "GET" and "limit=1" in query and "id=eq." not in query:
            return httpx.Response(200, json=[])

        # 2. Get session with messages: /rest/v1/chat_sessions?id=eq.<id>&select=*,chat_messages(*)
        if url_path == "/rest/v1/chat_sessions" and request.method == "GET" and "id=eq." in query:
            session_id = query.split("id=eq.")[1].split("&")[0]
            if session_id in self.chat_sessions:
                sess = dict(self.chat_sessions[session_id])
                # Join messages
                sess["chat_messages"] = [
                    m for m in self.chat_messages if m["session_id"] == session_id
                ]
                return httpx.Response(200, json=[sess])
            return httpx.Response(200, json=[])

        # 3. Upsert session: POST /rest/v1/chat_sessions?on_conflict=id
        if url_path == "/rest/v1/chat_sessions" and request.method == "POST":
            data = json.loads(request.content)
            session_id = data["id"]
            if session_id not in self.chat_sessions:
                self.chat_sessions[session_id] = {
                    "id": session_id,
                    "user_id": data.get("user_id"),
                    "language": data.get("language", "en"),
                    "created_at": data.get("created_at") or datetime.now(timezone.utc).isoformat(),
                    "updated_at": data.get("updated_at") or datetime.now(timezone.utc).isoformat()
                }
            else:
                self.chat_sessions[session_id].update({
                    "user_id": data.get("user_id", self.chat_sessions[session_id].get("user_id")),
                    "language": data.get("language", self.chat_sessions[session_id].get("language")),
                    "updated_at": data.get("updated_at") or datetime.now(timezone.utc).isoformat()
                })
            return httpx.Response(201, json=[self.chat_sessions[session_id]])

        # 4. Insert message: POST /rest/v1/chat_messages
        if url_path == "/rest/v1/chat_messages" and request.method == "POST":
            data = json.loads(request.content)
            self.chat_messages.append(data)
            return httpx.Response(201, json=[data])

        # 5. Get recent messages: GET /rest/v1/chat_messages?session_id=eq.<id>&order=created_at.desc&limit=<N>
        if url_path == "/rest/v1/chat_messages" and request.method == "GET":
            session_id = query.split("session_id=eq.")[1].split("&")[0]
            indexed_msgs = [(idx, m) for idx, m in enumerate(self.chat_messages) if m["session_id"] == session_id]
            indexed_msgs.sort(key=lambda x: (x[1].get("created_at", ""), x[0]), reverse=True)
            limit = 10
            if "limit=" in query:
                limit = int(query.split("limit=")[1].split("&")[0])
            matching = [x[1] for x in indexed_msgs[:limit]]
            return httpx.Response(200, json=matching)

        # 6. Delete session: DELETE /rest/v1/chat_sessions?id=eq.<id>
        if url_path == "/rest/v1/chat_sessions" and request.method == "DELETE":
            session_id = query.split("id=eq.")[1].split("&")[0]
            if session_id in self.chat_sessions:
                del self.chat_sessions[session_id]
            self.chat_messages = [m for m in self.chat_messages if m["session_id"] != session_id]
            return httpx.Response(204)

        return httpx.Response(404, json={"message": "Route not found"})


@pytest.fixture
def mock_db_state():
    state = PostgRESTMockState()
    return state


@pytest.fixture
def mock_supabase_repo(mock_db_state):
    transport = httpx.MockTransport(mock_db_state.handle_request)
    client = httpx.Client(transport=transport)
    repo = SupabaseChatRepository(
        supabase_url=MOCK_SUPABASE_URL,
        supabase_key=MOCK_SUPABASE_KEY,
        client=client
    )
    yield repo
    repo.close()


@pytest.fixture(autouse=True)
def cleanup_repo():
    yield
    reset_chat_repository()


# ---------------------------------------------------------------------------
# 1. Connection & Validation Tests
# ---------------------------------------------------------------------------

def test_supabase_repo_initialization():
    """Verify repository validation on empty or missing credentials."""
    with pytest.raises(ValueError):
        SupabaseChatRepository(supabase_url="", supabase_key="key")

    with pytest.raises(ValueError):
        SupabaseChatRepository(supabase_url="http://example.com", supabase_key="")


def test_supabase_verify_connection_success(mock_supabase_repo):
    """Verify health check returns True on valid 200 response."""
    assert mock_supabase_repo.verify_connection() is True


def test_supabase_verify_connection_auth_error():
    """Verify 401 status raises SupabaseConnectionError."""
    def handler(request):
        return httpx.Response(401, json={"message": "Invalid API key"})
    
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    repo = SupabaseChatRepository(MOCK_SUPABASE_URL, "bad-key", client=client)
    with pytest.raises(SupabaseConnectionError):
        repo.verify_connection()
    repo.close()


def test_supabase_verify_connection_table_missing():
    """Verify 404 status raises SupabaseQueryError indicating missing table."""
    def handler(request):
        return httpx.Response(404, json={"message": "Relation public.chat_sessions does not exist"})
    
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    repo = SupabaseChatRepository(MOCK_SUPABASE_URL, MOCK_SUPABASE_KEY, client=client)
    with pytest.raises(SupabaseQueryError):
        repo.verify_connection()
    repo.close()


# ---------------------------------------------------------------------------
# 2. CRUD Operations on Sessions and Messages
# ---------------------------------------------------------------------------

def test_supabase_session_crud(mock_supabase_repo):
    """Verify complete CRUD lifecycle of a chat session in Supabase."""
    session_id = "test_crud_session_001"
    
    # 1. Get non-existent session
    sess = mock_supabase_repo.get_session(session_id)
    assert sess is None

    # 2. Save session
    now = datetime.now(timezone.utc)
    new_sess = ChatSessionSchema(
        id=session_id,
        user_id="user_123",
        language="ta",
        created_at=now,
        updated_at=now,
        messages=[]
    )
    saved = mock_supabase_repo.save_session(new_sess)
    assert saved.id == session_id

    # 3. Retrieve session
    retrieved = mock_supabase_repo.get_session(session_id)
    assert retrieved is not None
    assert retrieved.id == session_id
    assert retrieved.user_id == "user_123"
    assert retrieved.language == "ta"

    # 4. Add message
    msg = ChatMessageSchema(
        session_id=session_id,
        role="user",
        message="வணக்கம்",
        language="ta",
        intent="greeting",
        module="chat"
    )
    added = mock_supabase_repo.add_message(session_id, msg)
    assert added.id is not None
    assert added.role == "user"

    # 5. Retrieve with messages
    retrieved_with_msgs = mock_supabase_repo.get_session(session_id)
    assert len(retrieved_with_msgs.messages) == 1
    assert retrieved_with_msgs.messages[0].message == "வணக்கம்"

    # 6. Delete session
    deleted = mock_supabase_repo.delete_session(session_id)
    assert deleted is True
    assert mock_supabase_repo.get_session(session_id) is None


# ---------------------------------------------------------------------------
# 3. Persistence across Backend Restart Simulation
# ---------------------------------------------------------------------------

def test_persistence_across_backend_restart(mock_db_state):
    """
    Simulates backend shutdown and restart:
    Instance 1 writes a session and messages.
    Instance 1 is destroyed and repository singleton is reset.
    Instance 2 boots up, connects to database, and retrieves the persisted session.
    """
    transport = httpx.MockTransport(mock_db_state.handle_request)
    
    # --- Backend Run 1 ---
    client1 = httpx.Client(transport=transport)
    repo1 = SupabaseChatRepository(MOCK_SUPABASE_URL, MOCK_SUPABASE_KEY, client=client1)
    set_chat_repository(repo1)

    sess_id = "restart_test_sess_001"
    session_service.get_or_create_session(session_id=sess_id, language="en")
    session_service.record_interaction(
        session_id=sess_id,
        user_message="My cow weighs 420 kg.",
        assistant_reply="Noted. Your cow weighs 420 kg. How much milk does it produce daily?",
        language="en",
        intent="nutrition",
        module="nutrition"
    )
    repo1.close()

    # --- Simulate Restart ---
    reset_chat_repository()

    # --- Backend Run 2 (Post-Restart) ---
    client2 = httpx.Client(transport=transport)
    repo2 = SupabaseChatRepository(MOCK_SUPABASE_URL, MOCK_SUPABASE_KEY, client=client2)
    set_chat_repository(repo2)

    history = session_service.get_history(sess_id)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].message == "My cow weighs 420 kg."
    assert history[1].role == "assistant"
    assert "420 kg" in history[1].message

    repo2.close()


# ---------------------------------------------------------------------------
# 4. Multi-Turn Conversational Persistence Test
# ---------------------------------------------------------------------------

def test_multiturn_conversation_persistence(mock_supabase_repo):
    """
    Tests 3-turn interactive conversation with context resolution from Supabase history:
    Turn 1: 'My cow weighs 420 kg.'
    Turn 2: 'It gives 15 litres of milk.'
    Turn 3: 'What should I feed it?'
    """
    set_chat_repository(mock_supabase_repo)
    sess_id = "multiturn_supabase_001"

    # Turn 1
    req1 = ChatRequest(message="My cow weighs 420 kg.", session_id=sess_id)
    res1 = chat_service.process_message(req1)
    assert res1.success is True
    assert res1.session_id == sess_id

    # Turn 2
    req2 = ChatRequest(message="It gives 15 litres of milk.", session_id=sess_id)
    res2 = chat_service.process_message(req2)
    assert res2.success is True

    # Turn 3
    req3 = ChatRequest(message="What should I feed it?", session_id=sess_id)
    res3 = chat_service.process_message(req3)
    assert res3.success is True
    assert res3.intent in ["nutrition", "feed"]

    # Verify that all 6 messages (3 user + 3 assistant) exist in repository
    history = mock_supabase_repo.get_recent_messages(sess_id, limit=20)
    assert len(history) == 6
    assert history[0].message == "My cow weighs 420 kg."
    assert history[2].message == "It gives 15 litres of milk."
    assert history[4].message == "What should I feed it?"


# ---------------------------------------------------------------------------
# 5. Multilingual Unicode Persistence Test (Tamil, Hindi, Tanglish)
# ---------------------------------------------------------------------------

def test_multilingual_unicode_persistence(mock_supabase_repo):
    """Verifies that native Indic scripts (Tamil, Hindi) and Tanglish persist without corruption."""
    set_chat_repository(mock_supabase_repo)

    # 1. Tamil Multi-turn
    sess_ta = "sess_tamil_unicode_001"
    req_ta1 = ChatRequest(message="என் மாடு 420 கிலோ எடை இருக்கு.", session_id=sess_ta)
    res_ta1 = chat_service.process_message(req_ta1)
    assert res_ta1.success is True
    assert res_ta1.language == "ta"

    req_ta2 = ChatRequest(message="தினமும் 15 லிட்டர் பால் தருது.", session_id=sess_ta)
    res_ta2 = chat_service.process_message(req_ta2)
    assert res_ta2.success is True

    history_ta = mock_supabase_repo.get_recent_messages(sess_ta)
    assert len(history_ta) >= 2
    assert "420 கிலோ" in history_ta[0].message
    assert "15 லிட்டர்" in history_ta[2].message

    # 2. Hindi Multi-turn
    sess_hi = "sess_hindi_unicode_001"
    req_hi = ChatRequest(message="मेरी गाय 15 लीटर दूध देती है, क्या खिलाएं?", session_id=sess_hi)
    res_hi = chat_service.process_message(req_hi)
    assert res_hi.success is True
    assert res_hi.language == "hi"

    history_hi = mock_supabase_repo.get_recent_messages(sess_hi)
    assert len(history_hi) >= 2
    assert "दूध देती है" in history_hi[0].message

    # 3. Tanglish (Romanized Tamil)
    sess_tanglish = "sess_tanglish_001"
    req_tg = ChatRequest(message="En maadu 15 litre paal kudukuthu, enna feed kudukkanum?", session_id=sess_tanglish)
    res_tg = chat_service.process_message(req_tg)
    assert res_tg.success is True
    assert res_tg.detected_language == "ta"


# ---------------------------------------------------------------------------
# 6. User Ownership & Security Access Control
# ---------------------------------------------------------------------------

def test_user_session_ownership_protection(mock_supabase_repo):
    """Verify that a session created by user A cannot be hijacked by user B."""
    set_chat_repository(mock_supabase_repo)
    sess_id = "protected_user_sess_001"

    # User A creates the session
    session_service.get_or_create_session(session_id=sess_id, user_id="farmer_alice_001")

    # User A accesses their own session - Should succeed
    sess_alice = session_service.get_or_create_session(session_id=sess_id, user_id="farmer_alice_001")
    assert sess_alice.user_id == "farmer_alice_001"

    # User B attempts to access User A's session - Should raise SessionAccessDeniedError
    with pytest.raises(SessionAccessDeniedError):
        session_service.get_or_create_session(session_id=sess_id, user_id="farmer_bob_999")


def test_api_user_session_ownership_endpoint(mock_supabase_repo):
    """Verify API returns 403 when unauthorized user accesses protected session."""
    set_chat_repository(mock_supabase_repo)
    sess_id = "api_protected_sess_001"

    with TestClient(app) as client:
        # Alice initiates chat
        res1 = client.post(
            "/api/v1/chat",
            json={"message": "Hello", "session_id": sess_id, "user_id": "alice_id"}
        )
        assert res1.status_code == 200

        # Bob attempts to use same session_id
        res2 = client.post(
            "/api/v1/chat",
            json={"message": "Hello from Bob", "session_id": sess_id, "user_id": "bob_id"}
        )
        assert res2.status_code == 403
