import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import VideoScriptRequest, VideoSocialMetadataRequest
from app.services import llm


class TestScriptPromptOptions(unittest.TestCase):
    def test_normalize_text_response_removes_think_blocks(self):
        """
        reasoning 模型可能返回 `<think>...</think>`。脚本生成链路必须只保留
        最终正文，避免思考过程进入字幕和配音。
        """
        result = llm._normalize_text_response(
            "<think>\nI should reason here.\n</think>\n测试成功",
            "minimax",
        )

        self.assertEqual(result, "测试成功")

    def test_normalize_text_response_rejects_think_only_response(self):
        """
        如果模型只返回思考块而没有最终答案，应视为空内容，触发重试或明确错误。
        """
        with self.assertRaises(ValueError):
            llm._normalize_text_response("<think>hidden reasoning</think>", "minimax")

    def test_normalize_text_response_removes_unclosed_think_block(self):
        """
        某些网关可能因为截断只返回未闭合的 `<think>`。这种内容同样不能
        进入最终脚本；如果清理后没有正文，就应该按空响应处理。
        """
        with self.assertRaises(ValueError):
            llm._normalize_text_response("<think>hidden reasoning", "minimax")

    def test_build_script_prompt_appends_advanced_requirements(self):
        """
        高级文案要求只作为附加约束，不替换默认系统提示词。
        这样普通用户不配置时仍然走稳定默认规则，高级用户也能细化风格。
        """
        prompt = llm.build_script_prompt(
            video_subject="咖啡",
            language="zh-CN",
            paragraph_number=3,
            video_script_prompt="语气轻松，面向程序员",
        )

        self.assertIn("# Role: Video Script Generator", prompt)
        self.assertIn("- video subject: 咖啡", prompt)
        self.assertIn("- number of paragraphs: 3", prompt)
        self.assertIn("- language: zh-CN", prompt)
        self.assertIn("# Additional User Requirements:", prompt)
        self.assertIn("语气轻松，面向程序员", prompt)

    def test_custom_system_prompt_keeps_runtime_context(self):
        """
        自定义 system prompt 会替换默认脚本规则，但视频主题、语言、段落数
        仍由服务层统一追加，避免高级用户漏写必要上下文。
        """
        prompt = llm.build_script_prompt(
            video_subject="露营",
            language="en",
            paragraph_number=2,
            custom_system_prompt="Only write cinematic narration.",
        )

        self.assertNotIn("# Role: Video Script Generator", prompt)
        self.assertIn("Only write cinematic narration.", prompt)
        self.assertIn("- video subject: 露营", prompt)
        self.assertIn("- number of paragraphs: 2", prompt)
        self.assertIn("- language: en", prompt)

    def test_generate_script_sends_custom_prompt_to_llm(self):
        captured = {}

        def fake_generate_response(prompt):
            captured["prompt"] = prompt
            return "第一段。\n\n第二段。"

        with patch.object(llm, "_generate_response", side_effect=fake_generate_response):
            result = llm.generate_script(
                video_subject="咖啡",
                language="zh-CN",
                paragraph_number=2,
                video_script_prompt="开头更有悬念",
            )

        self.assertEqual(result, "第一段。\n\n第二段。")
        self.assertIn("- number of paragraphs: 2", captured["prompt"])
        self.assertIn("开头更有悬念", captured["prompt"])

    def test_video_script_request_rejects_invalid_advanced_options(self):
        """
        API 请求模型需要限制高级 prompt 参数，避免外部调用绕过 WebUI
        传入异常段落数或超长提示词，导致模型成本和结果不可控。
        """
        with self.assertRaises(ValidationError):
            VideoScriptRequest(video_subject="咖啡", paragraph_number=0)

        with self.assertRaises(ValidationError):
            VideoScriptRequest(
                video_subject="咖啡",
                video_script_prompt="x" * (llm.MAX_SCRIPT_PROMPT_LENGTH + 1),
            )


class TestLiteLLMProvider(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def _use_litellm_provider(self, model_name="openai/gpt-4o-mini"):
        config.app["llm_provider"] = "litellm"
        config.app["litellm_model_name"] = model_name

    def test_litellm_provider_returns_normalized_text(self):
        """
        验证 LiteLLM provider 的主路径不依赖真实网络和私有 API key。

        这里用 fake module 注入 `sys.modules`，直接覆盖动态 import 的
        `litellm.completion()`，确保测试稳定覆盖 `_generate_response()` 里的
        litellm 分支。
        """
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace()

        def _completion(**kwargs):
            self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")
            self.assertEqual(
                kwargs["messages"], [{"role": "user", "content": "Say hello"}]
            )
            self.assertTrue(kwargs["drop_params"])
            message = types.SimpleNamespace(content="hello\nworld")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

        fake_litellm.completion = _completion

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "helloworld")

    def test_litellm_provider_requires_model_name(self):
        self._use_litellm_provider(model_name="")

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("model_name is not set", result)

    def test_litellm_provider_handles_empty_response(self):
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace(
            completion=lambda **kwargs: types.SimpleNamespace(choices=[])
        )

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("returned empty response", result)

    def test_litellm_provider_handles_empty_message(self):
        """
        某些 OpenAI-compatible 网关在内容过滤或安全拦截时会返回
        HTTP 200，但 `choices[0].message` 为 None。这里必须返回
        可诊断的错误，而不是抛出 AttributeError。
        """
        self._use_litellm_provider()

        fake_litellm = types.SimpleNamespace(
            completion=lambda **kwargs: types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=None)]
            )
        )

        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("returned empty message", result)

    def test_openai_provider_still_uses_existing_path(self):
        config.app["llm_provider"] = "openai"
        config.app["openai_api_key"] = ""
        config.app["openai_base_url"] = "https://api.openai.com/v1"
        config.app["openai_model_name"] = "gpt-4o-mini"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key is not set", result)
        self.assertNotIn("litellm", result.lower())

    def _use_qwen_provider(self):
        config.app["llm_provider"] = "qwen"
        config.app["qwen_api_key"] = "qwen-key"
        config.app["qwen_model_name"] = "qwen-max"

    def _patch_dashscope_generation(self, response):
        class FakeGenerationResponse(dict):
            pass

        fake_response = FakeGenerationResponse(response)
        fake_response.status_code = response.get("status_code", 200)
        fake_dashscope = types.SimpleNamespace(
            api_key="",
            Generation=types.SimpleNamespace(call=lambda **kwargs: fake_response),
        )
        fake_dashscope_response = types.SimpleNamespace(
            GenerationResponse=FakeGenerationResponse
        )

        return patch.dict(
            sys.modules,
            {
                "dashscope": fake_dashscope,
                "dashscope.api_entities": types.SimpleNamespace(),
                "dashscope.api_entities.dashscope_response": fake_dashscope_response,
            },
        )

    def test_qwen_provider_reads_chat_choices_content(self):
        """
        DashScope chat 模式会把文本放在 `output.choices[0].message.content`。
        这里覆盖 issue #966 报告的 `output.text is None` 场景，避免再次触发
        `'NoneType' object has no attribute 'replace'`。
        """
        self._use_qwen_provider()
        response = {
            "output": {
                "text": None,
                "choices": [{"message": {"content": "你好\n世界"}}],
            }
        }

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "你好世界")

    def test_qwen_provider_falls_back_to_output_text(self):
        """保留旧 DashScope completion 响应结构的兼容路径。"""
        self._use_qwen_provider()
        response = {"output": {"text": "旧格式\n响应"}}

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "旧格式响应")

    def test_qwen_provider_reports_empty_text(self):
        """Qwen 空响应应返回可诊断错误，而不是底层 AttributeError。"""
        self._use_qwen_provider()
        response = {"output": {"text": None, "choices": [{"message": {"content": None}}]}}

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertIn("Error:", result)
        self.assertIn("returned empty text content", result)
        self.assertNotIn("NoneType", result)

    def test_qwen_provider_reports_empty_choices(self):
        """Qwen chat 响应 choices 为空时应返回明确错误。"""
        self._use_qwen_provider()
        response = {"output": {"text": None, "choices": []}}

        with self._patch_dashscope_generation(response):
            result = llm._generate_response("Say hello")

        self.assertIn("Error:", result)
        self.assertIn("returned empty choices", result)
        self.assertNotIn("NoneType", result)

    def test_aihubmix_provider_uses_openai_compatible_client(self):
        """
        AIHubMix 是 OpenAI-compatible 网关。这里用 fake OpenAI client
        验证独立 provider 会使用合作方默认地址和推荐模型，避免真实网络
        或私有 API Key 影响测试稳定性。
        """
        config.app["llm_provider"] = "aihubmix"
        config.app["aihubmix_api_key"] = "aihubmix-key"
        config.app["aihubmix_base_url"] = ""
        config.app["aihubmix_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\naihubmix")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="aihubmix-key",
            base_url="https://aihubmix.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-5.4-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "helloaihubmix")

    def test_grok_provider_still_uses_existing_path(self):
        config.app["llm_provider"] = "grok"
        config.app["grok_api_key"] = ""
        config.app["grok_base_url"] = "https://api.x.ai/v1"
        config.app["grok_model_name"] = "grok-4.3"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key is not set", result)
        self.assertNotIn("litellm", result.lower())

    def test_groq_provider_requires_api_key(self):
        config.app["llm_provider"] = "groq"
        config.app["groq_api_key"] = ""
        config.app["groq_base_url"] = "https://api.groq.com/openai/v1"
        config.app["groq_model_name"] = "llama-3.3-70b-versatile"

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("api_key is not set", result)
        self.assertNotIn("litellm", result.lower())

    def test_groq_provider_uses_default_base_url(self):
        config.app["llm_provider"] = "groq"
        config.app["groq_api_key"] = "groq-test-key"
        config.app["groq_base_url"] = ""
        config.app["groq_model_name"] = "llama-3.3-70b-versatile"

        fake_response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="hello\ngroq")
                )
            ]
        )
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: fake_response)
            )
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="groq-test-key",
            base_url="https://api.groq.com/openai/v1",
        )
        self.assertEqual(result, "hellogroq")

    def _use_ollama_provider(self, base_url=""):
        config.app["llm_provider"] = "ollama"
        config.app["ollama_api_key"] = ""
        config.app["ollama_base_url"] = base_url
        config.app["ollama_model_name"] = "llama3"

    def _assert_ollama_base_url(self, expected_base_url: str):
        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nollama")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="ollama",
            base_url=expected_base_url,
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "llama3",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "helloollama")

    def test_ollama_default_base_url_uses_localhost_outside_container(self):
        """
        普通本机运行时，Ollama 默认仍然使用 localhost，避免影响已有用户。
        """
        self._use_ollama_provider()

        with patch.object(config, "is_running_in_container", return_value=False):
            self._assert_ollama_base_url("http://localhost:11434/v1")

    def test_ollama_default_base_url_uses_host_gateway_inside_container(self):
        """
        容器内运行时，localhost 指向容器自身；默认改为 host.docker.internal，
        方便 Docker Desktop 用户访问宿主机上的 Ollama。
        """
        self._use_ollama_provider()

        with (
            patch.object(config, "is_running_in_container", return_value=True),
            patch.object(config, "_can_resolve_hostname", return_value=True),
        ):
            self._assert_ollama_base_url("http://host.docker.internal:11434/v1")

    def test_ollama_default_base_url_falls_back_to_container_gateway(self):
        """
        原生 Linux Docker 里不一定能解析 host.docker.internal。此时使用容器
        默认网关作为兜底地址，比直接返回不可解析的 hostname 更稳。
        """
        self._use_ollama_provider()

        with (
            patch.object(config, "is_running_in_container", return_value=True),
            patch.object(config, "_can_resolve_hostname", return_value=False),
            patch.object(config, "get_container_default_gateway_ip", return_value="172.17.0.1"),
        ):
            self._assert_ollama_base_url("http://172.17.0.1:11434/v1")

    def test_ollama_explicit_base_url_takes_precedence(self):
        """
        用户手动配置的 ollama_base_url 优先级最高，不受容器检测影响。
        """
        self._use_ollama_provider(base_url="http://ollama:11434/v1")

        with patch.object(config, "is_running_in_container", return_value=True):
            self._assert_ollama_base_url("http://ollama:11434/v1")

    def test_mimo_provider_uses_openai_compatible_client(self):
        """
        MiMo 官方接口兼容 OpenAI Chat Completions 协议。这里用 fake OpenAI
        client 验证 provider 会使用 MiMo 独立配置和默认 base_url，不依赖
        真实网络或私有 API Key。
        """
        config.app["llm_provider"] = "mimo"
        config.app["mimo_api_key"] = "mimo-key"
        config.app["mimo_base_url"] = ""
        config.app["mimo_model_name"] = ""

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nmimo")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "OpenAI", return_value=fake_client) as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        openai_client.assert_called_once_with(
            api_key="mimo-key",
            base_url="https://api.xiaomimimo.com/v1",
        )
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "mimo-v2.5-pro",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "hellomimo")

    def test_azure_provider_uses_azure_client_directly(self):
        """
        Azure OpenAI 的鉴权、endpoint 和 api-version 都由 AzureOpenAI 客户端处理。
        这个测试覆盖 issue #892：azure 分支必须直接调用 AzureOpenAI 创建的客户端，
        不能继续落入普通 OpenAI-compatible 分支，否则会丢失 Azure 专用请求配置。
        """
        config.app["llm_provider"] = "azure"
        config.app["azure_api_key"] = "azure-key"
        config.app["azure_base_url"] = "https://example.openai.azure.com"
        config.app["azure_model_name"] = "gpt-4o-mini"
        config.app["azure_api_version"] = "2024-02-15-preview"

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                message = types.SimpleNamespace(content="hello\nazure")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_completions = FakeCompletions()
        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=fake_completions)
        )

        with (
            patch.object(llm, "AzureOpenAI", return_value=fake_client) as azure_client,
            patch.object(llm, "OpenAI") as openai_client,
            patch.object(llm, "ChatCompletion", types.SimpleNamespace),
        ):
            result = llm._generate_response("Say hello")

        azure_client.assert_called_once_with(
            api_key="azure-key",
            api_version="2024-02-15-preview",
            azure_endpoint="https://example.openai.azure.com",
        )
        openai_client.assert_not_called()
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(result, "helloazure")

    def test_g4f_provider_requires_explicit_opt_in(self):
        """
        g4f 存在供应链和稳定性风险，不能因为用户把 provider 写成 g4f
        就默认加载第三方包并访问逆向接口，必须显式启用。
        """
        config.app["llm_provider"] = "g4f"
        config.app["enable_g4f"] = False

        result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("g4f provider is disabled", result)

    def test_g4f_provider_uses_lazy_import_after_opt_in(self):
        config.app["llm_provider"] = "g4f"
        config.app["enable_g4f"] = True
        config.app["g4f_model_name"] = "gpt-3.5-turbo"

        fake_g4f = types.SimpleNamespace()
        fake_g4f.ChatCompletion = types.SimpleNamespace(
            create=lambda **kwargs: "hello from g4f"
        )

        with patch.dict(sys.modules, {"g4f": fake_g4f}):
            result = llm._generate_response("test")

        self.assertEqual(result, "hello from g4f")

    def test_g4f_provider_reports_missing_optional_dependency(self):
        config.app["llm_provider"] = "g4f"
        config.app["enable_g4f"] = True
        config.app["g4f_model_name"] = "gpt-3.5-turbo"

        with patch.dict(sys.modules, {"g4f": None}):
            result = llm._generate_response("test")

        self.assertIn("Error:", result)
        self.assertIn("g4f package is not installed by default", result)


class TestRuntimeEnvironmentDetection(unittest.TestCase):
    def test_container_detection_ignores_plain_linux_cgroup_file(self):
        """
        普通 Linux 也有 /proc/1/cgroup，不能因为文件存在就判定为容器。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            cgroup_path = Path(tmp_dir) / "cgroup"
            cgroup_path.write_text("0::/init.scope\n", encoding="utf-8")

            self.assertFalse(
                config.is_running_in_container(
                    dockerenv_path=str(Path(tmp_dir) / "missing-dockerenv"),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(cgroup_path),
                )
            )

    def test_container_detection_accepts_dockerenv_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dockerenv_path = Path(tmp_dir) / ".dockerenv"
            dockerenv_path.write_text("", encoding="utf-8")

            self.assertTrue(
                config.is_running_in_container(
                    dockerenv_path=str(dockerenv_path),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(Path(tmp_dir) / "missing-cgroup"),
                )
            )

    def test_container_detection_accepts_cgroup_container_marker(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cgroup_path = Path(tmp_dir) / "cgroup"
            cgroup_path.write_text(
                "0::/system.slice/docker-abcdef.scope\n",
                encoding="utf-8",
            )

            self.assertTrue(
                config.is_running_in_container(
                    dockerenv_path=str(Path(tmp_dir) / "missing-dockerenv"),
                    containerenv_path=str(Path(tmp_dir) / "missing-containerenv"),
                    cgroup_path=str(cgroup_path),
                )
            )

    def test_container_gateway_ip_decodes_default_route(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            route_path = Path(tmp_dir) / "route"
            route_path.write_text(
                "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
                "eth0\t00000000\t010011AC\t0003\t0\t0\t0\t00000000\t0\t0\t0\n",
                encoding="utf-8",
            )

            self.assertEqual(
                config.get_container_default_gateway_ip(str(route_path)),
                "172.17.0.1",
            )

    def test_container_gateway_ip_ignores_missing_default_route(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            route_path = Path(tmp_dir) / "route"
            route_path.write_text(
                "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
                "eth0\t0011AC0A\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n",
                encoding="utf-8",
            )

            self.assertEqual(config.get_container_default_gateway_ip(str(route_path)), "")


class TestSocialMetadata(unittest.TestCase):
    """通用短视频发布文案元数据生成。"""

    def test_build_prompt_auto_language_uses_source_language(self):
        """
        language 默认 auto 时，不应该固定成某个国家或语种，而是让模型
        跟随视频主题和脚本的语言，扩大 API 适用范围。
        """
        prompt = llm.build_social_metadata_prompt(
            video_subject="上海一日游",
            video_script="今天带你快速看完上海经典路线。",
            language="auto",
            platform="tiktok",
        )

        self.assertIn("TikTok", prompt)
        self.assertIn("Use the same language as the video subject and script", prompt)
        self.assertIn("上海一日游", prompt)
        self.assertIn("array of exactly 5 strings", prompt)

    def test_build_prompt_accepts_explicit_language(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="Coffee tips",
            language="en-US",
            platform="youtube_shorts",
        )

        self.assertIn("YouTube Shorts", prompt)
        self.assertIn('Write "title" and "caption" in this language: en-US', prompt)
        self.assertIn("array of exactly 3 strings", prompt)

    def test_unknown_platform_falls_back_to_tiktok(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="x",
            platform="unsupported-platform",
        )

        self.assertIn("TikTok", prompt)

    def test_normalize_hashtags_from_string_dedupes_and_clamps(self):
        tags = llm._normalize_hashtags("#fyp fyp, trending #Trending viral", count=2)

        self.assertEqual(tags, ["#fyp", "#trending"])

    def test_normalize_hashtags_from_list_keeps_unicode_letters(self):
        tags = llm._normalize_hashtags(
            ["上海 旅行", "#việt nam", "  ", "@bad!chars"], count=5
        )

        self.assertEqual(tags, ["#上海旅行", "#việtnam", "#badchars"])

    def test_parse_social_metadata_recovers_embedded_json(self):
        raw = 'Sure: {"title":"T","caption":"C","hashtags":["#x"]} thanks'
        result = llm._parse_social_metadata(raw, "tiktok")

        self.assertEqual(result["title"], "T")
        self.assertEqual(result["caption"], "C")
        self.assertEqual(result["hashtags"], ["#x"])

    def test_parse_social_metadata_requires_title_or_caption(self):
        with self.assertRaises(ValueError):
            llm._parse_social_metadata('{"hashtags":["#x"]}', "tiktok")

    def test_generate_social_metadata_uses_llm_response(self):
        payload = (
            '{"title":"上海一日游","caption":"收藏这条路线，下次直接出发！",'
            '"hashtags":["#上海","#旅行","#shorts"]}'
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            result = llm.generate_social_metadata(
                video_subject="上海一日游",
                video_script="今天带你快速看完上海经典路线。",
                language="zh-CN",
                platform="tiktok",
            )

        self.assertEqual(result["title"], "上海一日游")
        self.assertEqual(result["caption"], "收藏这条路线，下次直接出发！")
        self.assertEqual(result["hashtags"], ["#上海", "#旅行", "#shorts"])

    def test_generate_social_metadata_falls_back_to_generic_hashtags(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: api_key is not set"
        ):
            result = llm.generate_social_metadata(
                video_subject="Coffee tips",
                video_script="Save these three coffee tips.",
                platform="instagram_reels",
            )

        self.assertEqual(result["title"], "Coffee tips")
        self.assertEqual(result["caption"], "Save these three coffee tips.")
        self.assertEqual(len(result["hashtags"]), 8)
        self.assertEqual(result["hashtags"][0], "#shorts")

    def test_request_model_defaults_to_auto_language_tiktok(self):
        body = VideoSocialMetadataRequest(video_subject="Test")

        self.assertEqual(body.language, "auto")
        self.assertEqual(body.platform, "tiktok")

    def test_request_model_rejects_oversized_social_metadata_fields(self):
        """
        外部 API 不能接受无限长的脚本和语言参数，否则会直接放大 LLM
        token 成本。schema 层先拦截，服务层再做内部调用兜底。
        """
        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x" * 501)

        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x", video_script="x" * 8001)

        with self.assertRaises(ValidationError):
            VideoSocialMetadataRequest(video_subject="x", language="x" * 65)

    def test_build_prompt_clamps_direct_service_inputs(self):
        prompt = llm.build_social_metadata_prompt(
            video_subject="x" * 600,
            video_script="y" * 9000,
            language="en",
        )

        self.assertIn("x" * llm.MAX_SOCIAL_SUBJECT_LENGTH, prompt)
        self.assertNotIn("x" * (llm.MAX_SOCIAL_SUBJECT_LENGTH + 1), prompt)
        self.assertIn("y" * llm.MAX_SOCIAL_SCRIPT_LENGTH, prompt)
        self.assertNotIn("y" * (llm.MAX_SOCIAL_SCRIPT_LENGTH + 1), prompt)

    def test_social_metadata_endpoint_response_shape(self):
        from fastapi.testclient import TestClient

        from app.asgi import app

        request_body = {
            "video_subject": "Tokyo coffee shops",
            "video_script": "Three quiet coffee shops for your next Tokyo morning.",
            "language": "en",
            "platform": "youtube_shorts",
        }
        llm_response = (
            '{"title":"3 Quiet Tokyo Coffee Shops",'
            '"caption":"Save these spots for your next Tokyo morning.",'
            '"hashtags":["#Tokyo","#Coffee","#Shorts"]}'
        )

        with patch.object(llm, "_generate_response", return_value=llm_response):
            response = TestClient(app).post(
                "/api/v1/social-metadata",
                json=request_body,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": 200,
                "message": "success",
                "data": {
                    "title": "3 Quiet Tokyo Coffee Shops",
                    "caption": "Save these spots for your next Tokyo morning.",
                    "hashtags": ["#Tokyo", "#Coffee", "#Shorts"],
                },
            },
        )


class TestShotList(unittest.TestCase):
    """TikTok 带货分镜（shot list）生成。"""

    def test_generate_shot_list_parses_and_orders_shots(self):
        payload = json.dumps(
            [
                {
                    "scene": "Hold the product to camera",
                    "voiceover": "This changed my morning",
                    "onscreen_text": "WATCH THIS",
                    "broll": "",
                },
                {
                    "scene": "Close-up demo",
                    "voiceover": "Look how fast it works",
                    "onscreen_text": "so fast",
                    "broll": "kitchen gadget closeup",
                },
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            shots = llm.generate_shot_list(
                video_subject="mini blender",
                video_script="Make a smoothie in 30 seconds.",
                language="English",
            )

        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[0]["scene"], "Hold the product to camera")
        self.assertEqual(shots[1]["broll"], "kitchen gadget closeup")
        for shot in shots:
            self.assertEqual(set(shot.keys()), set(llm.SHOT_KEYS))

    def test_generate_shot_list_drops_items_without_scene(self):
        payload = json.dumps(
            [
                {"scene": "", "voiceover": "no scene -> dropped"},
                {"not": "a shot"},
                "not a dict",
                {"scene": "Real shot", "voiceover": "kept"},
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            shots = llm.generate_shot_list(video_subject="x")

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0]["scene"], "Real shot")

    def test_generate_shot_list_recovers_embedded_json(self):
        payload = 'Sure thing! [{"scene": "Shot one"}] hope that helps'
        with patch.object(llm, "_generate_response", return_value=payload):
            shots = llm.generate_shot_list(video_subject="x")

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0]["scene"], "Shot one")

    def test_generate_shot_list_clamps_amount(self):
        self.assertEqual(llm._normalize_shot_count(999), llm.MAX_SHOT_COUNT)
        self.assertEqual(llm._normalize_shot_count(0), 1)
        self.assertEqual(llm._normalize_shot_count("bad"), llm.DEFAULT_SHOT_COUNT)

    def test_generate_shot_list_returns_empty_on_error(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: api_key is not set"
        ):
            shots = llm.generate_shot_list(video_subject="x")

        self.assertEqual(shots, [])


class TestCommentReplies(unittest.TestCase):
    """TikTok 带货评论区回复 / 异议处理生成。"""

    def test_generate_comment_replies_parses_pairs(self):
        payload = json.dumps(
            [
                {"comment": "where can I buy this?", "reply": "Link in my bio!"},
                {
                    "comment": "does it actually work?",
                    "reply": "Yes — full demo is in the video.",
                },
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            replies = llm.generate_comment_replies(
                video_subject="mini blender", language="English"
            )

        self.assertEqual(len(replies), 2)
        self.assertEqual(replies[0]["comment"], "where can I buy this?")
        for pair in replies:
            self.assertEqual(set(pair.keys()), set(llm.COMMENT_REPLY_KEYS))

    def test_generate_comment_replies_drops_incomplete_pairs(self):
        payload = json.dumps(
            [
                {"comment": "too expensive"},
                {"reply": "no comment -> dropped"},
                "not a dict",
                {"comment": "price?", "reply": "Check the link in bio."},
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            replies = llm.generate_comment_replies(video_subject="x")

        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0]["comment"], "price?")

    def test_generate_comment_replies_clamps_amount(self):
        self.assertEqual(
            llm._normalize_comment_reply_count(999), llm.MAX_COMMENT_REPLY_COUNT
        )
        self.assertEqual(llm._normalize_comment_reply_count(0), 1)
        self.assertEqual(
            llm._normalize_comment_reply_count("bad"),
            llm.DEFAULT_COMMENT_REPLY_COUNT,
        )

    def test_generate_comment_replies_returns_empty_on_error(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: api_key is not set"
        ):
            replies = llm.generate_comment_replies(video_subject="x")

        self.assertEqual(replies, [])


class TestProductIdeas(unittest.TestCase):
    """TikTok 带货选品（product idea）生成。"""

    def test_generate_product_ideas_parses_objects(self):
        payload = json.dumps(
            [
                {
                    "product": "mini blender",
                    "category": "kitchen",
                    "reason": "easy to demo",
                    "audience": "busy people",
                    "angle": "30s smoothie",
                },
                {
                    "product": "LED strip",
                    "category": "home",
                    "reason": "satisfying reveal",
                    "audience": "students",
                    "angle": "room glow-up",
                },
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            ideas = llm.generate_product_ideas(
                category="gadgets", market="US", language="English"
            )

        self.assertEqual(len(ideas), 2)
        self.assertEqual(ideas[0]["product"], "mini blender")
        for idea in ideas:
            self.assertEqual(set(idea.keys()), set(llm.PRODUCT_IDEA_KEYS))

    def test_generate_product_ideas_drops_items_without_product(self):
        payload = json.dumps(
            [
                {"category": "kitchen", "reason": "no product -> dropped"},
                "not a dict",
                {"product": "real gadget", "category": "home"},
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            ideas = llm.generate_product_ideas()

        self.assertEqual(len(ideas), 1)
        self.assertEqual(ideas[0]["product"], "real gadget")

    def test_generate_product_ideas_clamps_amount(self):
        self.assertEqual(
            llm._normalize_product_idea_count(999), llm.MAX_PRODUCT_IDEA_COUNT
        )
        self.assertEqual(llm._normalize_product_idea_count(0), 1)
        self.assertEqual(
            llm._normalize_product_idea_count("bad"),
            llm.DEFAULT_PRODUCT_IDEA_COUNT,
        )

    def test_generate_product_ideas_returns_empty_on_error(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: api_key is not set"
        ):
            self.assertEqual(llm.generate_product_ideas(), [])


class TestHookVariations(unittest.TestCase):
    """TikTok 带货开场钩子（hook）生成。"""

    def test_generate_hook_variations_parses_strings(self):
        payload = json.dumps(
            [
                "Stop scrolling if your kitchen is a mess",
                "I wish someone told me this sooner",
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            hooks = llm.generate_hook_variations(
                video_subject="mini blender", language="English"
            )

        self.assertEqual(hooks, [
            "Stop scrolling if your kitchen is a mess",
            "I wish someone told me this sooner",
        ])

    def test_generate_hook_variations_strips_markers_and_dedupes(self):
        payload = json.dumps(
            [
                '1. "Stop scrolling now"',
                "- Stop scrolling now",
                "• A second hook",
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            hooks = llm.generate_hook_variations(video_subject="x")

        # The numbered/quoted and bulleted duplicates collapse to one entry.
        self.assertEqual(hooks, ["Stop scrolling now", "A second hook"])

    def test_generate_hook_variations_clamps_amount(self):
        self.assertEqual(llm._normalize_hook_count(999), llm.MAX_HOOK_COUNT)
        self.assertEqual(llm._normalize_hook_count(0), 1)
        self.assertEqual(llm._normalize_hook_count("bad"), llm.DEFAULT_HOOK_COUNT)

    def test_generate_hook_variations_returns_empty_on_error(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: api_key is not set"
        ):
            self.assertEqual(llm.generate_hook_variations(video_subject="x"), [])


class TestSoundIdeas(unittest.TestCase):
    """TikTok 带货配乐（trending sound）建议生成。"""

    def test_generate_sound_ideas_parses_objects(self):
        payload = json.dumps(
            [
                {
                    "sound": "upbeat pop",
                    "vibe": "energetic",
                    "search": "happy upbeat pop",
                    "tip": "sync the product reveal to the beat",
                },
                {
                    "sound": "satisfying ASMR",
                    "vibe": "calm and crisp",
                    "search": "asmr satisfying",
                    "tip": "use it over the close-up demo",
                },
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            ideas = llm.generate_sound_ideas(
                video_subject="mini blender", language="English"
            )

        self.assertEqual(len(ideas), 2)
        self.assertEqual(ideas[0]["sound"], "upbeat pop")
        for idea in ideas:
            self.assertEqual(set(idea.keys()), set(llm.SOUND_KEYS))

    def test_generate_sound_ideas_drops_items_without_sound(self):
        payload = json.dumps(
            [
                {"vibe": "energetic", "search": "no sound -> dropped"},
                "not a dict",
                {"sound": "calm aesthetic", "vibe": "chill"},
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            ideas = llm.generate_sound_ideas(video_subject="x")

        self.assertEqual(len(ideas), 1)
        self.assertEqual(ideas[0]["sound"], "calm aesthetic")

    def test_generate_sound_ideas_recovers_embedded_json(self):
        payload = 'Here you go: [{"sound": "suspense build-up"}] enjoy'
        with patch.object(llm, "_generate_response", return_value=payload):
            ideas = llm.generate_sound_ideas(video_subject="x")

        self.assertEqual(len(ideas), 1)
        self.assertEqual(ideas[0]["sound"], "suspense build-up")

    def test_generate_sound_ideas_clamps_amount(self):
        self.assertEqual(llm._normalize_sound_count(999), llm.MAX_SOUND_COUNT)
        self.assertEqual(llm._normalize_sound_count(0), 1)
        self.assertEqual(llm._normalize_sound_count("bad"), llm.DEFAULT_SOUND_COUNT)

    def test_generate_sound_ideas_returns_empty_on_error(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: api_key is not set"
        ):
            self.assertEqual(llm.generate_sound_ideas(video_subject="x"), [])


class TestTextStickers(unittest.TestCase):
    """TikTok 带货 on-screen text / sticker & CTA 生成。"""

    def test_generate_text_stickers_parses_objects(self):
        payload = json.dumps(
            [
                {
                    "text": "Wait for it...",
                    "timing": "first 2 seconds",
                    "style": "bold white top-center",
                    "purpose": "hook",
                },
                {
                    "text": "Tap the link in bio",
                    "timing": "at the end",
                    "style": "bold yellow, animated",
                    "purpose": "call-to-action",
                },
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            stickers = llm.generate_text_stickers(
                video_subject="mini blender", language="English"
            )

        self.assertEqual(len(stickers), 2)
        self.assertEqual(stickers[0]["text"], "Wait for it...")
        for sticker in stickers:
            self.assertEqual(set(sticker.keys()), set(llm.STICKER_KEYS))

    def test_generate_text_stickers_drops_items_without_text(self):
        payload = json.dumps(
            [
                {"timing": "intro", "style": "bold"},
                "not a dict",
                {"text": "Buy it now", "purpose": "cta"},
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            stickers = llm.generate_text_stickers(video_subject="x")

        self.assertEqual(len(stickers), 1)
        self.assertEqual(stickers[0]["text"], "Buy it now")

    def test_generate_text_stickers_recovers_embedded_json(self):
        payload = 'Sure: [{"text": "Only 3 left"}] done'
        with patch.object(llm, "_generate_response", return_value=payload):
            stickers = llm.generate_text_stickers(video_subject="x")

        self.assertEqual(len(stickers), 1)
        self.assertEqual(stickers[0]["text"], "Only 3 left")

    def test_generate_text_stickers_clamps_amount(self):
        self.assertEqual(llm._normalize_sticker_count(999), llm.MAX_STICKER_COUNT)
        self.assertEqual(llm._normalize_sticker_count(0), 1)
        self.assertEqual(
            llm._normalize_sticker_count("bad"), llm.DEFAULT_STICKER_COUNT
        )

    def test_generate_text_stickers_returns_empty_on_error(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: api_key is not set"
        ):
            self.assertEqual(llm.generate_text_stickers(video_subject="x"), [])


class TestCoverTextIdeas(unittest.TestCase):
    """TikTok 带货 cover / thumbnail 标题生成。"""

    def test_generate_cover_text_parses_objects(self):
        payload = json.dumps(
            [
                {
                    "text": "I was wrong about this",
                    "subtext": "honest review",
                    "angle": "curiosity",
                    "tip": "use a surprised-face frame",
                },
                {
                    "text": "Under $20?!",
                    "subtext": "",
                    "angle": "benefit",
                    "tip": "bright high-contrast frame",
                },
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            ideas = llm.generate_cover_text_ideas(
                video_subject="mini blender", language="English"
            )

        self.assertEqual(len(ideas), 2)
        self.assertEqual(ideas[0]["text"], "I was wrong about this")
        for idea in ideas:
            self.assertEqual(set(idea.keys()), set(llm.COVER_KEYS))

    def test_generate_cover_text_drops_items_without_text(self):
        payload = json.dumps(
            [
                {"subtext": "no headline", "angle": "curiosity"},
                "not a dict",
                {"text": "3 reasons I switched", "angle": "social proof"},
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            ideas = llm.generate_cover_text_ideas(video_subject="x")

        self.assertEqual(len(ideas), 1)
        self.assertEqual(ideas[0]["text"], "3 reasons I switched")

    def test_generate_cover_text_recovers_embedded_json(self):
        payload = 'Options: [{"text": "Don\'t buy until you see this"}] ok'
        with patch.object(llm, "_generate_response", return_value=payload):
            ideas = llm.generate_cover_text_ideas(video_subject="x")

        self.assertEqual(len(ideas), 1)
        self.assertEqual(ideas[0]["text"], "Don't buy until you see this")

    def test_generate_cover_text_clamps_amount(self):
        self.assertEqual(llm._normalize_cover_count(999), llm.MAX_COVER_COUNT)
        self.assertEqual(llm._normalize_cover_count(0), 1)
        self.assertEqual(llm._normalize_cover_count("bad"), llm.DEFAULT_COVER_COUNT)

    def test_generate_cover_text_returns_empty_on_error(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: api_key is not set"
        ):
            self.assertEqual(llm.generate_cover_text_ideas(video_subject="x"), [])


class TestPostingSchedule(unittest.TestCase):
    """TikTok 带货发布时间建议生成。"""

    def test_generate_posting_schedule_parses_objects(self):
        payload = json.dumps(
            [
                {
                    "slot": "Prime evening",
                    "day": "Weekdays",
                    "time": "7:00-9:00 PM",
                    "why": "shoppers browse after work",
                },
                {
                    "slot": "Lunch break",
                    "day": "Every day",
                    "time": "12:00-1:00 PM",
                    "why": "midday scroll",
                },
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            slots = llm.generate_posting_schedule(
                video_subject="mini blender",
                language="English",
                audience_region="Vietnam",
            )

        self.assertEqual(len(slots), 2)
        self.assertEqual(slots[0]["slot"], "Prime evening")
        for slot in slots:
            self.assertEqual(set(slot.keys()), set(llm.SCHEDULE_KEYS))

    def test_generate_posting_schedule_drops_items_without_time(self):
        payload = json.dumps(
            [
                {"slot": "no time", "day": "Mon", "why": "dropped"},
                "not a dict",
                {"slot": "Evening", "time": "8 PM"},
            ]
        )
        with patch.object(llm, "_generate_response", return_value=payload):
            slots = llm.generate_posting_schedule(video_subject="x")

        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["time"], "8 PM")

    def test_generate_posting_schedule_recovers_embedded_json(self):
        payload = 'Here: [{"time": "9:00-10:00 PM"}] thanks'
        with patch.object(llm, "_generate_response", return_value=payload):
            slots = llm.generate_posting_schedule(video_subject="x")

        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["time"], "9:00-10:00 PM")

    def test_generate_posting_schedule_clamps_amount(self):
        self.assertEqual(llm._normalize_schedule_count(999), llm.MAX_SCHEDULE_COUNT)
        self.assertEqual(llm._normalize_schedule_count(0), 1)
        self.assertEqual(
            llm._normalize_schedule_count("bad"), llm.DEFAULT_SCHEDULE_COUNT
        )

    def test_generate_posting_schedule_returns_empty_on_error(self):
        with patch.object(
            llm, "_generate_response", return_value="Error: api_key is not set"
        ):
            self.assertEqual(llm.generate_posting_schedule(video_subject="x"), [])


FOUNDRY_KEY = os.environ.get("ANTHROPIC_FOUNDRY_API_KEY", "")
FOUNDRY_BASE = "https://amanrai-test-resource.services.ai.azure.com/anthropic"
FOUNDRY_MODEL = "azure_ai/claude-sonnet-4-6"


@unittest.skipUnless(FOUNDRY_KEY, "ANTHROPIC_FOUNDRY_API_KEY not set")
class TestLiteLLMLiveIntegration(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["llm_provider"] = "litellm"
        config.app["litellm_model_name"] = FOUNDRY_MODEL
        os.environ["AZURE_AI_API_KEY"] = FOUNDRY_KEY
        os.environ["AZURE_AI_API_BASE"] = FOUNDRY_BASE

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_live_litellm_completion(self):
        result = llm._generate_response("What is 2+2? Reply with just the number.")

        self.assertNotIn("Error:", result)
        self.assertIn("4", result)


if __name__ == "__main__":
    unittest.main()
