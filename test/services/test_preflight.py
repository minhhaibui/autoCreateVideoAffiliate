import unittest

from app.services.preflight import (
    is_llm_ready,
    llm_provider_requires_key,
)


class TestLlmProviderRequiresKey(unittest.TestCase):
    def test_key_providers_require_a_key(self):
        for provider in ["openai", "gemini", "deepseek", "azure", "aihubmix"]:
            self.assertTrue(llm_provider_requires_key(provider))

    def test_keyless_providers_do_not_require_a_key(self):
        for provider in ["ollama", "g4f", "pollinations"]:
            self.assertFalse(llm_provider_requires_key(provider))

    def test_is_case_and_whitespace_insensitive(self):
        self.assertFalse(llm_provider_requires_key("  Ollama  "))
        self.assertTrue(llm_provider_requires_key("OpenAI"))

    def test_empty_provider_treated_as_needing_a_key(self):
        # An unknown/empty provider should fail safe: assume a key is needed so
        # the readiness banner shows rather than silently letting a call fail.
        self.assertTrue(llm_provider_requires_key(""))
        self.assertTrue(llm_provider_requires_key(None))


class TestIsLlmReady(unittest.TestCase):
    def test_key_provider_ready_only_with_key(self):
        self.assertFalse(is_llm_ready("openai", ""))
        self.assertFalse(is_llm_ready("openai", "   "))
        self.assertFalse(is_llm_ready("openai", None))
        self.assertTrue(is_llm_ready("openai", "sk-abc123"))

    def test_keyless_provider_always_ready(self):
        self.assertTrue(is_llm_ready("ollama", ""))
        self.assertTrue(is_llm_ready("pollinations", None))
        self.assertTrue(is_llm_ready("g4f", ""))

    def test_key_is_stripped_before_checking(self):
        self.assertTrue(is_llm_ready("gemini", "  AQ.key  "))


if __name__ == "__main__":
    unittest.main()
