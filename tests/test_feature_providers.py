import pytest

from app.config import FEATURE_PROVIDERS, Settings

GATEWAY = {"ai_provider_base_url": "http://gateway.local/v1", "ai_provider_api_key": "gateway-key"}


def settings(**values):
    return Settings(_env_file=None, ai_model="qwen3:4b", **GATEWAY, **values)


def block(feature, **extra):
    values = {
        f"{feature}_provider_protocol": "openai",
        f"{feature}_provider_base_url": "https://api.openai.example/v1",
        f"{feature}_provider_api_key": f"{feature}-key",
        f"{feature}_model": "gpt-6-luna",
    }
    values.update(extra)
    return values


@pytest.mark.parametrize("feature", FEATURE_PROVIDERS)
def test_each_feature_stays_on_the_gateway_unless_fully_configured(feature):
    values = block(feature)
    current = settings()
    assert current.for_feature(feature) is current
    for missing in (f"{feature}_provider_api_key", f"{feature}_model"):
        partial = {k: v for k, v in values.items() if k != missing}
        current = settings(**partial)
        assert current.for_feature(feature) is current


@pytest.mark.parametrize("feature", FEATURE_PROVIDERS)
def test_each_feature_block_is_scoped_and_leaves_the_main_provider_alone(feature):
    current = settings(**block(feature))
    scoped = current.for_feature(feature)
    assert scoped.ai_provider_protocol == "openai"
    assert scoped.ai_model == "gpt-6-luna"
    assert scoped.ai_provider_api_key == f"{feature}-key"
    assert scoped.ai_provider_message_max_chars is None
    assert scoped.request_timeout_seconds == getattr(current, f"{feature}_timeout_seconds")
    assert scoped.ai_max_tokens == getattr(current, f"{feature}_max_tokens")
    assert current.ai_model == "qwen3:4b"
    assert str(current.ai_provider_base_url) == "http://gateway.local/v1"
    for other in FEATURE_PROVIDERS:
        if other != feature:
            assert current.for_feature(other) is current


def test_unknown_feature_is_refused():
    with pytest.raises(ValueError):
        settings().for_feature("document")


def test_discovery_alias_and_default_protocols():
    current = settings(**block("discovery", discovery_provider_protocol="anthropic"))
    assert current.for_discovery().ai_provider_protocol == "anthropic"
    assert settings().assistant_provider_protocol == "openai"
    assert settings().chat_provider_protocol == "openai"
    assert settings().chat_open_drafting is False
