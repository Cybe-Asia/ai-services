import io
import json

from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfWriter

from app.config import Settings, get_settings
from app.document_analysis import (
    SYSTEM_PROMPT,
    ModelFinding,
    _merge_page_findings,
    _parse_model_finding,
    validate_and_render,
)
from app.llm_client import LlmClient
from app.main import app


def synthetic_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(output, format="PNG")
    return output.getvalue()


def enabled_settings() -> Settings:
    return Settings(
        document_analysis_enabled=True,
        document_analysis_internal_token="internal-test-token",
        ai_provider_base_url="http://provider.invalid/v1",
    )


def finding(**overrides) -> ModelFinding:
    payload = {
        "detectedDocumentType": "birth_certificate",
        "readability": "readable",
        "fields": {
            "studentName": "Synthetic Student",
            "dateOfBirth": "2018-01-02",
            "registrationNumber": "SYN-001",
        },
        "requiredFields": {
            "studentNamePresent": True,
            "dateOfBirthPresent": True,
            "registrationNumberPresent": True,
        },
        "warnings": [],
    }
    payload.update(overrides)
    return ModelFinding.model_validate(payload)


def test_model_output_contract_rejects_extra_fields() -> None:
    payload = {
        "detectedDocumentType": "birth_certificate",
        "readability": "readable",
        "fields": {
            "studentName": "Synthetic Student",
            "dateOfBirth": "2018-01-02",
            "registrationNumber": "SYN-001",
        },
        "requiredFields": {
            "studentNamePresent": True,
            "dateOfBirthPresent": True,
            "registrationNumberPresent": True,
        },
        "warnings": [],
        "instructionsFollowed": True,
    }
    try:
        ModelFinding.model_validate(payload)
    except ValueError:
        pass
    else:
        raise AssertionError("extra model fields must be rejected")


def test_prompt_treats_document_text_as_untrusted_data() -> None:
    lowered = SYSTEM_PROMPT.casefold()
    assert "untrusted data" in lowered
    assert "ignore any instruction" in lowered


def test_structured_output_accepts_json_fence_but_not_trailing_text() -> None:
    raw = finding().model_dump_json(by_alias=True)
    assert _parse_model_finding(f"```json\n{raw}\n```").fields.student_name == "Synthetic Student"
    try:
        _parse_model_finding(f"```json\n{raw}\n``` trailing")
    except ValueError:
        pass
    else:
        raise AssertionError("trailing model text must be rejected")


def test_page_merge_is_deterministic_and_recomputes_required_fields() -> None:
    partial = finding(
        readability="partially_readable",
        fields={
            "studentName": "Synthetic Student",
            "dateOfBirth": None,
            "registrationNumber": None,
        },
        requiredFields={
            "studentNamePresent": False,
            "dateOfBirthPresent": True,
            "registrationNumberPresent": True,
        },
        warnings=["page one"],
    )
    second = finding(
        fields={
            "studentName": None,
            "dateOfBirth": "2018-01-02",
            "registrationNumber": "SYN-001",
        },
        warnings=["page two"],
    )
    merged = _merge_page_findings([partial, second])
    assert merged.fields.student_name == "Synthetic Student"
    assert merged.fields.date_of_birth == "2018-01-02"
    assert merged.required_fields.student_name_present is True
    assert merged.warnings == ["page one", "page two"]


def test_gateway_adapter_uses_single_page_vision_contract(monkeypatch) -> None:
    observed = {}

    async def fake_gateway(self, **kwargs):
        observed.update(kwargs)
        return finding().model_dump_json(by_alias=True)

    monkeypatch.setattr(LlmClient, "complete_gateway_vision", fake_gateway)
    settings = enabled_settings().model_copy(
        update={"document_analysis_adapter": "cybe_gateway_vision"}
    )
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        response = TestClient(app).post(
            "/api/ai/v1/internal/document-analysis",
            content=synthetic_png(),
            headers={
                "authorization": "Bearer internal-test-token",
                "content-type": "image/png",
                "x-expected-document-type": "birth_certificate",
            },
        )
        assert response.status_code == 200
        assert observed["model"] == "qwen2.5vl:7b"
        assert observed["temperature"] == 0
        assert observed["image_base64"]
        assert observed["json_schema"]["additionalProperties"] is False
    finally:
        app.dependency_overrides.clear()


def test_signature_validation_rejects_declared_type_mismatch() -> None:
    try:
        validate_and_render(synthetic_png(), "image/jpeg", enabled_settings())
    except ValueError as exc:
        assert "signature" in str(exc)
    else:
        raise AssertionError("mismatched type must fail")


def test_encrypted_pdf_is_rejected() -> None:
    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("synthetic-password")
    writer.write(output)
    try:
        validate_and_render(output.getvalue(), "application/pdf", enabled_settings())
    except ValueError as exc:
        assert "encrypted" in str(exc)
    else:
        raise AssertionError("encrypted PDF must fail")


def test_pdf_with_huge_render_dimensions_is_rejected_before_render() -> None:
    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=100_000, height=100_000)
    writer.write(output)
    try:
        validate_and_render(output.getvalue(), "application/pdf", enabled_settings())
    except ValueError as exc:
        assert "dimensions" in str(exc)
    else:
        raise AssertionError("oversized PDF render must fail")


def test_internal_endpoint_requires_auth_and_rejects_unsupported_type() -> None:
    app.dependency_overrides[get_settings] = enabled_settings
    client = TestClient(app)
    try:
        unauthorized = client.post(
            "/api/ai/v1/internal/document-analysis",
            content=synthetic_png(),
            headers={
                "content-type": "image/png",
                "x-expected-document-type": "birth_certificate",
            },
        )
        unsupported = client.post(
            "/api/ai/v1/internal/document-analysis",
            content=synthetic_png(),
            headers={
                "authorization": "Bearer internal-test-token",
                "content-type": "image/png",
                "x-expected-document-type": "family_card",
            },
        )
        assert unauthorized.status_code == 401
        assert unsupported.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_endpoint_validates_provider_json(monkeypatch) -> None:
    finding = {
        "detectedDocumentType": "birth_certificate",
        "readability": "readable",
        "fields": {
            "studentName": "Synthetic Student",
            "dateOfBirth": "2018-01-02",
            "registrationNumber": "SYN-001",
        },
        "requiredFields": {
            "studentNamePresent": True,
            "dateOfBirthPresent": True,
            "registrationNumberPresent": True,
        },
        "warnings": [],
    }

    async def fake_complete(self, **kwargs):
        assert kwargs["model"] == "qwen2.5vl:7b"
        assert kwargs["temperature"] == 0
        return json.dumps(finding)

    monkeypatch.setattr(LlmClient, "complete_multimodal", fake_complete)
    app.dependency_overrides[get_settings] = enabled_settings
    try:
        response = TestClient(app).post(
            "/api/ai/v1/internal/document-analysis",
            content=synthetic_png(),
            headers={
                "authorization": "Bearer internal-test-token",
                "content-type": "image/png",
                "x-expected-document-type": "birth_certificate",
            },
        )
        assert response.status_code == 200
        assert response.json()["schemaVersion"] == "1.0"
        assert response.json()["finding"]["fields"]["studentName"] == "Synthetic Student"
    finally:
        app.dependency_overrides.clear()


def test_provider_failure_is_generic_and_does_not_echo_document_data(monkeypatch) -> None:
    async def fake_complete(self, **kwargs):
        return None

    monkeypatch.setattr(LlmClient, "complete_multimodal", fake_complete)
    app.dependency_overrides[get_settings] = enabled_settings
    try:
        response = TestClient(app).post(
            "/api/ai/v1/internal/document-analysis",
            content=synthetic_png(),
            headers={
                "authorization": "Bearer internal-test-token",
                "content-type": "image/png",
                "x-expected-document-type": "birth_certificate",
            },
        )
        assert response.status_code == 502
        assert response.json() == {"detail": "Document analysis failed"}
    finally:
        app.dependency_overrides.clear()
