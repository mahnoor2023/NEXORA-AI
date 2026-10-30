from fastapi.testclient import TestClient
import json
import pytest
import pandas as pd

from app import agent
from app.agent import AgentError
from app.main import app
from app import database
from app.state import forget_user_training_results, remember_training_result, retrieve_training_result

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "nexora-test.sqlite3")
    monkeypatch.setattr(database, "STORAGE_ROOT", tmp_path / "dataset-files")
    database.initialize()
    client.cookies.clear()
    _signup(client, "api-tests@example.test")


def _signup(client_: TestClient, email: str, name: str = "Test User") -> None:
    response = client_.post("/api/v1/auth/signup", json={"email": email, "display_name": name, "password": "Correct-Horse-7!"})
    assert response.status_code == 200, response.text


def test_signup_login_logout_and_password_validation() -> None:
    user = TestClient(app)
    response = user.post("/api/v1/auth/signup", json={"email": "user@example.test", "display_name": "Test User", "password": "Correct-Horse-7!"})
    assert response.status_code == 200
    assert "nexora_session" in response.cookies
    assert user.get("/api/v1/auth/me").json()["user"]["email"] == "user@example.test"
    assert user.post("/api/v1/auth/signup", json={"email": "USER@example.test", "display_name": "Other", "password": "Correct-Horse-7!"}).status_code == 409
    assert user.post("/api/v1/auth/logout").status_code == 200
    assert user.get("/api/v1/auth/me").status_code == 401
    assert user.post("/api/v1/auth/login", json={"email": "user@example.test", "password": "Correct-Horse-7!"}).status_code == 200
    assert user.post("/api/v1/auth/login", json={"email": "user@example.test", "password": "wrong"}).status_code == 401


def test_dataset_metadata_is_persisted_and_isolated_between_users(tmp_path) -> None:
    first, second = TestClient(app), TestClient(app)
    _signup(first, "first@example.test")
    _signup(second, "second@example.test")
    file = {"file": ("private.csv", b"value\n1\n2\n", "text/csv")}
    result = first.post("/api/v1/datasets/analyze", files=file)
    assert result.status_code == 200, result.text
    dataset_id = result.json()["dataset_id"]
    assert len(first.get("/api/v1/datasets").json()["datasets"]) == 1
    assert second.get("/api/v1/datasets").json()["datasets"] == []
    assert second.get(f"/api/v1/datasets/{dataset_id}/analysis").status_code == 404
    assert second.get(f"/api/v1/datasets/{dataset_id}/source").status_code == 404
    assert second.delete(f"/api/v1/datasets/{dataset_id}").status_code == 404
    db_path = database.DB_PATH
    reopened = __import__("sqlite3").connect(db_path)
    assert reopened.execute("SELECT COUNT(*) FROM analysis_history").fetchone()[0] == 1
    reopened.close()
    # A new application client sees the persisted user/session after the first client is gone.
    restarted_client = TestClient(app)
    restarted_client.cookies.update(first.cookies)
    assert restarted_client.get("/api/v1/auth/me").status_code == 200
    assert len(restarted_client.get("/api/v1/datasets").json()["datasets"]) == 1
    assert restarted_client.get(f"/api/v1/datasets/{dataset_id}/source").content == b"value\n1\n2\n"


def test_saved_csv_reopens_with_real_analysis_and_supports_training() -> None:
    owner = TestClient(app)
    _signup(owner, "reopen@example.test")
    source = _classification_csv()
    analyzed = owner.post("/api/v1/datasets/analyze", files={"file": ("classification.csv", source, "text/csv")})
    assert analyzed.status_code == 200, analyzed.text
    dataset_id = analyzed.json()["dataset_id"]
    reopened = owner.get(f"/api/v1/datasets/{dataset_id}/analysis")
    assert reopened.status_code == 200, reopened.text
    assert reopened.json()["filename"] == "classification.csv"
    assert reopened.json()["preview"] == analyzed.json()["preview"]
    saved_csv = owner.get(f"/api/v1/datasets/{dataset_id}/source")
    assert saved_csv.status_code == 200
    assert saved_csv.content == source
    trained = owner.post("/api/v1/datasets/train", files={"file": ("classification.csv", saved_csv.content, "text/csv")}, data={"target": "outcome", "task": "auto"})
    assert trained.status_code == 200, trained.text
    result = trained.json()
    assert result["task"] == "classification"
    assert len(result["models"]) == 2
    assert "f1_weighted" in result["models"][0]["metrics"]


def test_cleaning_never_changes_persisted_original_csv() -> None:
    raw = b"group,value\nA,2\nA,2\n,8\n"
    response = client.post("/api/v1/datasets/analyze", files={"file": ("clean.csv", raw, "text/csv")}, data={"cleaning": '{"drop_duplicates":true,"fill_categorical":true}'})
    assert response.status_code == 200
    user_id = client.get("/api/v1/auth/me").json()["user"]["id"]
    saved, _, metadata = database.dataset_source(user_id, response.json()["dataset_id"])
    assert saved == raw
    assert metadata["row_count"] == 3


def test_dataset_quota_rejects_upload_without_writing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(database, "MAX_DATASET_STORAGE_BYTES", 4)
    raw = b"value\n123\n"
    response = client.post("/api/v1/datasets/analyze", files={"file": ("large.csv", raw, "text/csv")})
    assert response.status_code == 413
    assert not list(database.STORAGE_ROOT.glob("*.csv"))


def test_dataset_deletion_removes_file_and_analysis_history() -> None:
    owner = TestClient(app)
    _signup(owner, "delete-dataset@example.test")
    result = owner.post("/api/v1/datasets/analyze", files={"file": ("remove.csv", b"x\n1\n", "text/csv")}).json()
    user_id = owner.get("/api/v1/auth/me").json()["user"]["id"]
    _, _, metadata = database.dataset_source(user_id, result["dataset_id"])
    path = database._storage_path(metadata["storage_key"])
    assert path.is_file()
    assert owner.delete(f"/api/v1/datasets/{result['dataset_id']}").status_code == 200
    assert not path.exists()
    assert owner.get("/api/v1/datasets/history").json()["history"] == []


def test_dataset_history_returns_sqlalchemy_row_mappings() -> None:
    uploaded = client.post("/api/v1/datasets/analyze", files={"file": ("history.csv", b"x,y\n1,2\n3,4\n", "text/csv")})
    assert uploaded.status_code == 200
    response = client.get("/api/v1/datasets/history")
    assert response.status_code == 200
    history = response.json()["history"]
    assert len(history) == 1
    assert history[0]["filename"] == "history.csv"
    assert history[0]["summary"]["row_count"] == 2


def test_account_deletion_removes_private_source_files() -> None:
    owner = TestClient(app)
    _signup(owner, "delete-account-files@example.test")
    result = owner.post("/api/v1/datasets/analyze", files={"file": ("account.csv", b"x\n1\n", "text/csv")}).json()
    user_id = owner.get("/api/v1/auth/me").json()["user"]["id"]
    _, _, metadata = database.dataset_source(user_id, result["dataset_id"])
    path = database._storage_path(metadata["storage_key"])
    assert path.is_file()
    deleted = owner.request("DELETE", "/api/v1/auth/account", json={"password": "Correct-Horse-7!", "confirmation": "DELETE"})
    assert deleted.status_code == 200
    assert not path.exists()
    with database.connect() as db:
        assert db.execute("SELECT COUNT(*) AS total FROM users WHERE id=?", (user_id,)).fetchone()["total"] == 0


def test_legacy_metadata_migrates_without_claiming_csv_is_available(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = tmp_path / "legacy.sqlite3"
    connection = __import__("sqlite3").connect(legacy)
    connection.executescript("""
        CREATE TABLE users(id TEXT PRIMARY KEY,email TEXT NOT NULL UNIQUE,display_name TEXT NOT NULL,password_hash TEXT NOT NULL,password_salt TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
        CREATE TABLE datasets(id TEXT PRIMARY KEY,owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,content_hash TEXT NOT NULL,filename TEXT NOT NULL,size_bytes INTEGER NOT NULL,row_count INTEGER NOT NULL,column_count INTEGER NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,UNIQUE(owner_id,content_hash),UNIQUE(id,owner_id));
    """)
    connection.execute("INSERT INTO users VALUES('legacy-user','legacy@example.test','Legacy','hash','salt','now','now')")
    connection.execute("INSERT INTO datasets VALUES('legacy-dataset','legacy-user','deadbeef','old.csv',12,2,1,'now','now')")
    connection.commit()
    connection.close()
    monkeypatch.setattr(database, "DB_PATH", legacy)
    database.initialize()
    with database.connect() as migrated:
        record = migrated.execute("SELECT * FROM datasets WHERE id='legacy-dataset'").fetchone()
    assert record["available"] == 0
    assert record["storage_key"] is None
    with pytest.raises(database.DatasetUnavailableError, match="metadata only"):
        database.dataset_source("legacy-user", "legacy-dataset")


def test_protected_routes_and_cross_origin_writes_are_rejected() -> None:
    anonymous = TestClient(app)
    assert anonymous.post("/api/v1/datasets/analyze", files={"file": ("a.csv", b"x\n1\n", "text/csv")}).status_code == 401
    denied = anonymous.post("/api/v1/auth/signup", headers={"Origin": "https://attacker.invalid"}, json={"email": "origin@example.test", "display_name": "Origin", "password": "Correct-Horse-7!"})
    assert denied.status_code == 403


def test_cross_site_cookie_configuration_is_applied_and_requires_secure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COOKIE_SECURE", "true")
    monkeypatch.setenv("COOKIE_SAMESITE", "none")
    user = TestClient(app, base_url="https://testserver")
    response = user.post("/api/v1/auth/signup", json={"email": "cookie@example.test", "display_name": "Cookie", "password": "Correct-Horse-7!"})
    assert response.status_code == 200
    cookie = response.headers["set-cookie"].lower()
    assert "secure" in cookie and "httponly" in cookie and "samesite=none" in cookie
    assert user.get("/api/v1/auth/me").status_code == 200


def test_storage_identifiers_are_uuid_scoped_and_local_objects_are_private(tmp_path) -> None:
    from app.storage import LocalStorage
    store = LocalStorage(tmp_path / "objects")
    key = "a1dc2e54-5d0b-4de3-9d80-c20c4d7ea2db"
    store.put(key, b"secret,csv\n1,2\n")
    assert store.get(key) == b"secret,csv\n1,2\n"
    assert list((tmp_path / "objects").iterdir()) == [tmp_path / "objects" / "a1dc2e545d0b4de39d80c20c4d7ea2db.csv"]
    with pytest.raises(KeyError):
        store.get("../../outside")
    store.delete(key)
    assert not list((tmp_path / "objects").iterdir())


def test_s3_storage_uses_private_uuid_object_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3
    from app.storage import S3Storage

    class MissingObject(Exception):
        pass

    class Exceptions:
        NoSuchKey = MissingObject

    class Body:
        def __init__(self, content: bytes):
            self.content = content
            self.closed = False
        def read(self):
            return self.content
        def close(self):
            self.closed = True

    class FakeS3:
        exceptions = Exceptions
        def __init__(self):
            self.objects = {}
            self.body = None
        def put_object(self, **kwargs):
            self.objects[(kwargs["Bucket"], kwargs["Key"])] = kwargs["Body"]
            assert "ACL" not in kwargs
        def get_object(self, **kwargs):
            content = self.objects.get((kwargs["Bucket"], kwargs["Key"]))
            if content is None:
                raise MissingObject()
            self.body = Body(content)
            return {"Body": self.body}
        def delete_object(self, **kwargs):
            self.objects.pop((kwargs["Bucket"], kwargs["Key"]), None)

    fake = FakeS3()
    monkeypatch.setattr(boto3, "client", lambda *_args, **_kwargs: fake)
    store = S3Storage("private-bucket", "auto", "https://object.example.test")
    key = "a1dc2e54-5d0b-4de3-9d80-c20c4d7ea2db"
    store.put(key, b"a,b\n1,2\n")
    assert ("private-bucket", "datasets/a1dc2e545d0b4de39d80c20c4d7ea2db.csv") in fake.objects
    assert store.get(key) == b"a,b\n1,2\n"
    assert fake.body.closed
    store.delete(key)
    with pytest.raises(KeyError):
        store.get(key)
    with pytest.raises(KeyError):
        store.get("../../outside")


def test_health_fails_closed_when_database_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable():
        raise RuntimeError("secret connection string")
    monkeypatch.setattr(database, "connect", unavailable)
    response = client.get("/api/v1/health")
    assert response.status_code == 503
    assert response.json() == {"status": "error", "database": "unavailable"}
    assert "secret" not in response.text


def test_account_cleanup_removes_transient_training_cache() -> None:
    raw = b"unique-cache-account-test"
    remember_training_result(raw, {"owner_id": "deleted-user", "best_model": "baseline"})
    assert retrieve_training_result(raw) is not None
    forget_user_training_results("deleted-user")
    assert retrieve_training_result(raw) is None


def test_settings_preferences_password_and_account_cleanup() -> None:
    user = TestClient(app)
    _signup(user, "settings@example.test")
    assert user.patch("/api/v1/auth/profile", json={"display_name": "Updated Name"}).json()["user"]["display_name"] == "Updated Name"
    prefs = {"default_view": "datasets", "compact_tables": True}
    assert user.patch("/api/v1/auth/preferences", json=prefs).json()["preferences"] == prefs
    assert user.get("/api/v1/auth/preferences").json()["preferences"] == prefs
    assert user.post("/api/v1/auth/password", json={"current_password": "bad", "new_password": "Long-New-Password-123"}).status_code == 422
    assert user.post("/api/v1/auth/password", json={"current_password": "Correct-Horse-7!", "new_password": "Long-New-Password-123"}).status_code == 200
    assert user.get("/api/v1/auth/me").status_code == 200
    assert user.post("/api/v1/auth/login", json={"email": "settings@example.test", "password": "Correct-Horse-7!"}).status_code == 401
    assert user.post("/api/v1/auth/login", json={"email": "settings@example.test", "password": "Long-New-Password-123"}).status_code == 200
    assert user.request("DELETE", "/api/v1/auth/account", json={"password": "Long-New-Password-123", "confirmation": "no"}).status_code == 422
    assert user.request("DELETE", "/api/v1/auth/account", json={"password": "Long-New-Password-123", "confirmation": "DELETE"}).status_code == 200
    assert user.get("/api/v1/auth/me").status_code == 401


@pytest.fixture(autouse=True)
def keep_insights_tests_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_API_KEY", "LLM_API_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_health() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_real_csv_statistics_and_preview() -> None:
    csv = b"name,revenue,region\nA,10,East\nB,,West\nC,30,East\n"
    response = client.post("/api/v1/datasets/analyze", files={"file": ("sample.csv", csv, "text/csv")})
    assert response.status_code == 200
    result = response.json()
    assert result["row_count"] == 3
    assert result["column_count"] == 3
    assert result["total_missing"] == 1
    assert result["columns"][1]["numeric"]["mean"] == 20
    assert result["preview"][1]["revenue"] is None


def test_rejects_non_csv_and_malformed_cleaning() -> None:
    assert client.post("/api/v1/datasets/analyze", files={"file": ("data.json", b"{}", "application/json")}).status_code == 415
    files = {"file": ("data.csv", b"a\n1\n", "text/csv")}
    assert client.post("/api/v1/datasets/analyze", files=files, data={"cleaning": "{}"}).status_code == 200
    for invalid in ("{", '{"unknown_option":true}', '{"fill_numeric":"yes"}'):
        response = client.post("/api/v1/datasets/analyze", files=files, data={"cleaning": invalid})
        assert response.status_code == 422


def test_constant_numeric_column_has_a_histogram() -> None:
    response = client.post("/api/v1/datasets/analyze", files={"file": ("constant.csv", b"value\n4\n4\n4\n", "text/csv")})
    assert response.status_code == 200
    assert response.json()["columns"][0]["histogram"] == [{"bin": "4", "count": 3}]


def test_cleaning_returns_derived_frame_and_preserves_missing_when_disabled() -> None:
    csv = b"group,value\nA,2\nA,2\n,8\n"
    base = client.post("/api/v1/datasets/analyze", files={"file": ("data.csv", csv, "text/csv")}).json()
    cleaned = client.post("/api/v1/datasets/analyze", files={"file": ("data.csv", csv, "text/csv")}, data={"cleaning": '{"drop_duplicates":true,"fill_numeric":true,"fill_categorical":true}'}).json()
    assert base["row_count"] == 3 and base["total_missing"] == 1
    assert cleaned["row_count"] == 2 and cleaned["total_missing"] == 0


def _classification_csv(include_target_copy: bool = False) -> bytes:
    header = "age,segment,outcome" + (",outcome_copy" if include_target_copy else "")
    rows = [header]
    for index in range(80):
        outcome = "yes" if index >= 40 else "no"
        age = "" if index == 5 else str(index)
        segment = "" if index == 7 else ("north" if index % 2 else "south")
        rows.append(f"{age},{segment},{outcome}" + (f",{outcome}" if include_target_copy else ""))
    return ("\n".join(rows) + "\n").encode()


def _regression_csv(rows_count: int = 80) -> bytes:
    rows = ["experience,department,salary"]
    for index in range(rows_count):
        department = "sales" if index % 2 else "engineering"
        rows.append(f"{index},{department},{2 * index + 5}")
    return ("\n".join(rows) + "\n").encode()


def test_train_classification_with_missing_values_and_leakage_safe_pipeline() -> None:
    files = {"file": ("employees.csv", _classification_csv(), "text/csv")}
    response = client.post("/api/v1/datasets/train", files=files, data={"target": "outcome", "task": "auto"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["task"] == "classification"
    assert result["random_seed"] == 42
    assert result["target"] == "outcome"
    assert result["features"] == ["age", "segment"]
    assert len(result["models"]) == 2
    assert "f1_weighted" in result["models"][0]["metrics"]
    assert result["models"][0]["confusion_matrix"] is not None
    assert result["insights"]["provider"] == "rule_based"
    assert "not LLM-generated" in result["insights"]["label"]


def test_train_regression_auto_detection_and_metrics() -> None:
    response = client.post(
        "/api/v1/datasets/train",
        files={"file": ("salaries.csv", _regression_csv(), "text/csv")},
        data={"target": "salary", "task": "auto"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["task"] == "regression"
    assert len(result["models"]) == 2
    assert {"mae", "rmse", "r2"}.issubset(result["models"][0]["metrics"])
    assert result["models"][0]["predictions"]


def test_task_can_be_manually_overridden_for_numeric_target() -> None:
    rows = ["signal,target"] + [f"{index},{int(index >= 40)}" for index in range(80)]
    response = client.post(
        "/api/v1/datasets/train",
        files={"file": ("binary.csv", ("\n".join(rows) + "\n").encode(), "text/csv")},
        data={"target": "target", "task": "regression"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["task"] == "regression"


def test_training_rejects_invalid_target_task_and_small_dataset() -> None:
    files = {"file": ("data.csv", _regression_csv(12), "text/csv")}
    small = client.post("/api/v1/datasets/train", files=files, data={"target": "salary", "task": "auto"})
    assert small.status_code == 422
    invalid_target = client.post(
        "/api/v1/datasets/train",
        files={"file": ("data.csv", _regression_csv(), "text/csv")},
        data={"target": "absent", "task": "auto"},
    )
    assert invalid_target.status_code == 422
    invalid_task = client.post(
        "/api/v1/datasets/train",
        files={"file": ("data.csv", _regression_csv(), "text/csv")},
        data={"target": "salary", "task": "clustering"},
    )
    assert invalid_task.status_code == 422


def test_training_rejects_exact_target_leakage_column() -> None:
    response = client.post(
        "/api/v1/datasets/train",
        files={"file": ("leaky.csv", _classification_csv(include_target_copy=True), "text/csv")},
        data={"target": "outcome", "task": "classification"},
    )
    assert response.status_code == 422
    assert "Potential target leakage" in response.json()["detail"]


def test_agent_tool_dispatch_calculates_real_dataset_evidence() -> None:
    frame = pd.DataFrame({"value": [1, 2, None, 4], "group": ["a", "a", "b", "b"]})
    schema = agent.dispatch_tool("inspect_dataset_schema", {}, frame, "values.csv")
    stats = agent.dispatch_tool("calculate_column_statistics", {"column": "value"}, frame, "values.csv")
    missing = agent.dispatch_tool("analyze_missing_values", {}, frame, "values.csv")
    assert schema["rows"] == 4 and schema["columns_count"] == 2
    assert stats["statistics"]["mean"] == pytest.approx(2.333333)
    assert missing["total_missing"] == 1


def test_agent_grounding_requires_real_tool_citations_and_numbers() -> None:
    evidence = {"inspect_dataset_schema": [{"rows": 40, "columns_count": 2}]}
    valid = agent.validate_grounded_response(
        {"answer": "The CSV has 40 rows.", "citations": [{"tool": "inspect_dataset_schema", "quote": '"rows": 40'}]},
        evidence,
    )
    assert "40 rows" in valid["answer"]
    with pytest.raises(AgentError, match="numeric claims"):
        agent.validate_grounded_response(
            {"answer": "The CSV has 4,000 rows.", "citations": [{"tool": "inspect_dataset_schema", "quote": '"rows": 40'}]},
            evidence,
        )
    with pytest.raises(AgentError, match="citation"):
        agent.validate_grounded_response(
            {"answer": "There are 40 rows.", "citations": [{"tool": "inspect_dataset_schema", "quote": '"rows": 41'}]},
            evidence,
        )


def test_agent_rejects_invalid_tool_arguments() -> None:
    frame = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(AgentError, match="Invalid argument names"):
        agent.dispatch_tool("inspect_dataset_schema", {"python": "print(1)"}, frame, "x.csv")
    with pytest.raises(AgentError, match="top_n"):
        agent.dispatch_tool("calculate_column_statistics", {"column": "value", "top_n": 11}, frame, "x.csv")
    with pytest.raises(AgentError, match="unknown tool"):
        agent.dispatch_tool("execute_python", {}, frame, "x.csv")


def test_agent_returns_explicit_ollama_unavailable_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "local-test-model")
    monkeypatch.setattr(agent, "_request_json", lambda *args, **kwargs: (_ for _ in ()).throw(AgentError("OLLAMA_UNAVAILABLE", "Ollama offline")))
    response = client.get("/api/v1/agent/status")
    assert response.status_code == 200
    assert response.json()["state"] == "ollama_unavailable"
    chat = client.post("/api/v1/agent/chat", files={"file": ("data.csv", b"x,y\n1,2\n", "text/csv")}, data={"question": "Describe this dataset"})
    assert chat.status_code == 503
    assert chat.json()["detail"]["code"] == "OLLAMA_UNAVAILABLE"


def test_agent_reports_model_unavailable_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "missing-local-model")
    monkeypatch.setattr(agent, "_request_json", lambda *args, **kwargs: {"models": [{"name": "installed-model:latest"}]})
    response = client.get("/api/v1/agent/status")
    assert response.status_code == 200
    assert response.json()["state"] == "model_unavailable"
    assert "not installed" in response.json()["message"]


def test_agent_loop_enforces_maximum_tool_call_limit() -> None:
    frame = pd.DataFrame({"value": range(20), "other": range(20, 40)})
    calls = 0

    def repeating_tool_call(_base: str, _payload: dict[str, object], _timeout: float) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"message": {"role": "assistant", "tool_calls": [{"function": {"name": "inspect_dataset_schema", "arguments": {}}}]}}

    with pytest.raises(AgentError, match="tool calls") as error:
        agent.run_data_analyst_agent("Inspect the file", frame, "x.csv", None, "local", "http://localhost:11434", chat_call=repeating_tool_call)
    assert error.value.code == "TOOL_CALL_LIMIT"
    assert calls == agent.MAX_AGENT_TURNS


def test_agent_loop_dispatches_real_tools_and_validates_grounded_answer() -> None:
    frame = pd.DataFrame({"value": range(40), "other": range(40, 80)})
    responses: list[dict[str, object]] = [
        {"message": {"role": "assistant", "tool_calls": [{"function": {"name": "inspect_dataset_schema", "arguments": {}}}]}},
        {"message": {"role": "assistant", "tool_calls": [{"function": {"name": "generate_evidence_based_summary", "arguments": {}}}]}},
        {"message": {"role": "assistant", "content": '{"answer":"The uploaded CSV has 40 rows.","citations":[{"tool":"generate_evidence_based_summary","quote":"\\"rows\\": 40"}]}'}},
    ]

    def local_protocol_stub(_base: str, _payload: dict[str, object], _timeout: float) -> dict[str, object]:
        return responses.pop(0)

    result = agent.run_data_analyst_agent("How many rows?", frame, "x.csv", None, "test-model", "http://localhost:11434", chat_call=local_protocol_stub)
    assert result["status"] == "complete"
    assert result["tools_used"] == ["inspect_dataset_schema", "generate_evidence_based_summary"]
    assert result["tool_call_count"] == 2


def test_groq_status_reports_missing_credentials_and_ready_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    missing = agent.groq_status()
    assert missing["state"] == "missing_credentials"
    monkeypatch.setenv("GROQ_API_KEY", "test-secret")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setattr(agent, "_groq_request", lambda *_args, **_kwargs: {"data": [{"id": "openai/gpt-oss-20b"}]})
    ready = agent.groq_status()
    assert ready["state"] == "ready"
    assert "third-party" in ready["message"]


def test_groq_status_distinguishes_rate_limit_and_failed_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-secret")
    monkeypatch.setattr(agent, "_groq_request", lambda *_args, **_kwargs: (_ for _ in ()).throw(AgentError("GROQ_RATE_LIMITED", "limited")))
    assert agent.groq_status()["state"] == "rate_limited"
    monkeypatch.setattr(agent, "_groq_request", lambda *_args, **_kwargs: (_ for _ in ()).throw(AgentError("GROQ_REQUEST_FAILED", "denied")))
    assert agent.groq_status()["state"] == "request_failed"


def test_groq_tool_call_conversation_uses_call_ids_and_bounded_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-secret")
    frame = pd.DataFrame({"sensitive_value": ["secret-cell-a", "secret-cell-b"], "score": [1, 2]})
    evidence = {"generate_evidence_based_summary": [{"rows": 2, "columns_count": 2}]}
    final_content = json.dumps({"answer": "The CSV has 2 rows.", "citations": [{"tool": "generate_evidence_based_summary", "quote": '"rows": 2'}]})
    responses = [
        {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call_stats_1", "type": "function", "function": {"name": "calculate_column_statistics", "arguments": "{\"column\":\"sensitive_value\"}"}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call_summary_2", "type": "function", "function": {"name": "generate_evidence_based_summary", "arguments": "{}"}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": final_content}}]},
    ]
    payloads: list[dict[str, object]] = []

    def fake_provider_response(_url: str, payload: dict[str, object], _timeout: float) -> dict[str, object]:
        payloads.append(payload)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_groq_request", fake_provider_response)
    result = agent.run_data_analyst_agent("How many rows?", frame, "private.csv", None, "openai/gpt-oss-20b", agent.GROQ_BASE_URL, chat_call=agent._groq_chat, provider="groq")
    assert result["provider"] == "groq"
    assert result["tool_call_count"] == 2
    second_messages = payloads[1]["messages"]
    assert any(message.get("role") == "tool" and message.get("tool_call_id") == "call_stats_1" for message in second_messages)
    assert "secret-cell-a" not in json.dumps(payloads)
    assert payloads[0]["tools"] == agent.TOOL_DEFINITIONS
    assert payloads[0]["tool_choice"] == "required"
    assert payloads[0]["parallel_tool_calls"] is False
    assert payloads[2]["response_format"]["json_schema"]["strict"] is True
