import asyncio

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app import learning_client
from app.config import Settings


@pytest.mark.parametrize(
    "origin",
    [
        "http://foreign.example",
        "http://learning-api",
        "http://learning-api.learning-platform.svc.cluster.local",
        "https://user:secret@school.example",
        "https://school.example/path",
        "https://school.example?token=x",
        "https://school.example#x",
        "file:///tmp/owner",
        "https://school.example:999999",
    ],
)
def test_learning_owner_configuration_cannot_forward_student_token_to_unsafe_origin(origin):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, learning_service_url=origin)


@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:3222",
        "https://learning-api.learning-platform.svc.cluster.local",
        "https://learning-api.example",
    ],
)
def test_secure_and_private_learning_service_origins_are_supported(origin):
    assert Settings(_env_file=None, learning_service_url=origin).learning_service_url == origin


@pytest.mark.parametrize(
    "authorization", [None, "Bearer ", "Basic x", "Bearer a b", "Bearer " + "x" * 16385]
)
def test_token_validation_precedes_network(monkeypatch, authorization):
    def no_client(*args, **kwargs):
        raise AssertionError("Invalid session must not reach an owner")

    monkeypatch.setattr(learning_client.httpx, "AsyncClient", no_client)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            learning_client.learning_data(
                Settings(_env_file=None), authorization, "/api/v1/learning/context"
            )
        )
    assert error.value.status_code == 401


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (401, {}, 401),
        (403, {}, 403),
        (404, {}, 404),
        (503, {}, 503),
        (302, {}, 503),
        (200, {"data": []}, 503),
        (200, {"error": "internal"}, 503),
    ],
)
def test_owner_errors_and_malformed_success_are_not_answers(monkeypatch, status, body, expected):
    original = httpx.AsyncClient

    def response(request):
        assert request.headers["Authorization"] == "Bearer student-session"
        return httpx.Response(status, json=body, headers={"Location": "https://foreign.example"})

    def client(**kwargs):
        assert kwargs["follow_redirects"] is False
        return original(**kwargs, transport=httpx.MockTransport(response))

    monkeypatch.setattr(learning_client.httpx, "AsyncClient", client)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            learning_client.learning_data(
                Settings(_env_file=None), "Bearer student-session", "/api/v1/learning/context"
            )
        )
    assert error.value.status_code == expected


def test_oversized_owner_response_fails_closed(monkeypatch):
    original = httpx.AsyncClient

    def client(**kwargs):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 1048577))
        return original(**kwargs, transport=transport)

    monkeypatch.setattr(learning_client.httpx, "AsyncClient", client)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            learning_client.learning_data(
                Settings(_env_file=None), "Bearer student-session", "/api/v1/learning/context"
            )
        )
    assert error.value.status_code == 503


@pytest.mark.parametrize("encoding", ["gzip", "br", "identity, gzip"])
def test_compressed_owner_body_is_rejected_before_decoding(monkeypatch, encoding):
    original = httpx.AsyncClient

    class EncodedStream(httpx.AsyncByteStream):
        read = False
        closed = False

        async def __aiter__(self):
            self.read = True
            yield b"must never decode this body"

        async def aclose(self):
            self.closed = True

    stream = EncodedStream()

    def response(request):
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, headers={"Content-Encoding": encoding}, stream=stream)

    monkeypatch.setattr(
        learning_client.httpx,
        "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(response)),
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            learning_client.learning_data(
                Settings(_env_file=None), "Bearer student-session", "/api/v1/learning/context"
            )
        )
    assert error.value.status_code == 503
    assert not stream.read
    assert stream.closed
