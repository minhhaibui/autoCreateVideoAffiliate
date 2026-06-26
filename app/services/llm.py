import json
import logging
import re
import requests
from typing import List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config

_max_retries = 5
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_DEPRECATED_GEMINI_MODELS = {"gemini-pro", "gemini-1.0-pro"}
MIN_SCRIPT_PARAGRAPH_NUMBER = 1
MAX_SCRIPT_PARAGRAPH_NUMBER = 10
MAX_SCRIPT_PROMPT_LENGTH = 2000
MAX_SCRIPT_SYSTEM_PROMPT_LENGTH = 8000
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)

DEFAULT_SCRIPT_SYSTEM_PROMPT = """
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.
""".strip()


# Specialized system prompt for TikTok affiliate / product-promotion videos.
# It reuses the same hard output constraints as the default prompt (plain
# spoken text, no markdown, same language as the subject) but steers the model
# toward a hook → benefits → call-to-action structure that converts viewers
# into buyers, while explicitly forbidding fabricated prices/stats so the
# output stays compliant with affiliate advertising rules.
TIKTOK_AFFILIATE_SCRIPT_SYSTEM_PROMPT = """
# Role: TikTok Affiliate Video Script Writer

## Goals:
Write a short, high-converting spoken script for a TikTok affiliate video that
promotes the product or topic given as the video subject and makes viewers want
to buy it through the affiliate link.

## Script flow (weave these into one smooth narration, never label the parts):
1. Hook: open with a scroll-stopping first line within the first 3 seconds - a
   bold claim, a relatable problem, or a surprising benefit.
2. Problem or desire: name the pain point or desire the product addresses.
3. Benefits: give two or three concrete, specific benefits framed as outcomes
   the viewer will get.
4. Light proof or urgency: hint that it is popular or a limited deal, WITHOUT
   inventing exact numbers, prices, or statistics.
5. Call to action: end by telling viewers to tap the product link or cart and
   buy it now.

## Constrains:
1. return the script as a single block of plain spoken text with the specified number of paragraphs.
2. use a spoken, energetic, conversational tone with short, punchy sentences that are easy to read aloud and fit on subtitles.
3. do not include any markdown, titles, emojis, hashtags, bullet points, or section labels.
4. do not include "voiceover", "narrator", stage directions, or similar indicators.
5. do not invent fake prices, fake discounts, exact statistics, or unverifiable medical or financial claims.
6. do not under any circumstance reference this prompt or the script itself.
7. only return the raw spoken content of the script.
8. respond in the same language as the video subject.
""".strip()


# Selectable script "styles" surfaced in the WebUI. A value of "" means use the
# default system prompt; any other value is passed through as a custom system
# prompt to build_script_prompt(), which still appends the runtime context
# (subject, language, paragraph count) for us.
SCRIPT_STYLE_PRESETS = {
    "default": "",
    "tiktok_affiliate": TIKTOK_AFFILIATE_SCRIPT_SYSTEM_PROMPT,
}


def get_script_style_system_prompt(style_key: str) -> str:
    """Return the system prompt for a named script style, or "" for the default
    style / any unknown key (callers then fall back to DEFAULT_SCRIPT_SYSTEM_PROMPT)."""
    return SCRIPT_STYLE_PRESETS.get(style_key or "default", "")


def _normalize_text_response(content, llm_provider: str) -> str:
    # 不同 LLM SDK 在异常或被拦截场景下，可能返回 None、空字符串，
    # 甚至返回非字符串对象。这里统一做兜底校验，避免后续直接调用
    # `.replace()` 时抛出 `NoneType` 之类的属性错误。
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    # MiniMax M3、DeepSeek R1 这类 reasoning 模型可能会把内部推理包在
    # `<think>...</think>` 中返回。视频脚本和关键词只需要最终可朗读文本，
    # 如果不在服务层统一清理，WebUI、字幕和配音都会把思考过程当正文处理。
    content = _THINK_BLOCK_RE.sub("", content)
    content = _UNCLOSED_THINK_BLOCK_RE.sub("", content).strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    return content.replace("\n", "")


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # OpenAI 兼容接口在异常场景下，可能返回没有 choices、
    # 或者 choices/message/content 为空的响应对象。
    # 这里统一做结构校验，避免出现 `NoneType is not subscriptable`
    # 这类底层属性访问错误。
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _get_response_field(value, key: str):
    """兼容 dict 和 SDK 响应对象的字段读取。"""
    if isinstance(value, dict):
        return value.get(key)

    try:
        return value[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(value, key, None)


def _extract_qwen_generation_text(response) -> str:
    """
    从 DashScope Generation 响应中提取文本。

    Qwen 使用 `messages` 调用时返回的是 chat 结构：
    `output.choices[0].message.content`；旧 completion 形态才会返回
    `output.text`。这里两个路径都兼容，避免 `output.text` 为 None 时
    继续 `.replace()` 触发不可诊断的 AttributeError。
    """
    output = _get_response_field(response, "output")
    choices = _get_response_field(output, "choices") if output else None
    if choices is not None:
        if not choices:
            logger.warning("Qwen returned an empty choices list")
            raise ValueError("[qwen] returned empty choices")

        first_choice = choices[0]
        message = _get_response_field(first_choice, "message")
        content = _get_response_field(message, "content") if message else None
        if content is not None:
            return _normalize_text_response(content, "qwen")

    text = _get_response_field(output, "text") if output else None
    return _normalize_text_response(text, "qwen")


def _generate_response(prompt: str) -> str:
    try:
        content = ""
        llm_provider = config.app.get("llm_provider", "openai")
        logger.info(f"llm provider: {llm_provider}")
        if llm_provider == "g4f":
            if not config.app.get("enable_g4f", False):
                raise ValueError(
                    "g4f provider is disabled by default because it relies on "
                    "reverse-engineered third-party endpoints. Set enable_g4f=true "
                    "in config.toml only if you understand and accept the security, "
                    "reliability, and legal risks."
                )

            logger.warning(
                "g4f provider is enabled. This provider may be unstable and carries "
                "supply-chain and terms-of-service risks. Prefer official providers, "
                "OpenAI-compatible APIs, LiteLLM, Ollama, or local inference for production."
            )
            try:
                import g4f
            except ImportError as e:
                raise ValueError(
                    "g4f package is not installed by default. Install the optional "
                    "dependency with `uv sync --extra g4f` only if you understand "
                    "and accept the provider risks."
                ) from e

            model_name = config.app.get("g4f_model_name", "")
            if not model_name:
                model_name = "gpt-3.5-turbo-16k-0613"
            content = g4f.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            api_version = ""  # for azure
            if llm_provider == "moonshot":
                api_key = config.app.get("moonshot_api_key")
                model_name = config.app.get("moonshot_model_name")
                base_url = "https://api.moonshot.cn/v1"
            elif llm_provider == "ollama":
                # api_key = config.app.get("openai_api_key")
                api_key = "ollama"  # any string works but you are required to have one
                model_name = config.app.get("ollama_model_name")
                base_url = config.app.get("ollama_base_url", "")
                if not base_url:
                    base_url = config.get_default_ollama_base_url()
            elif llm_provider == "openai":
                api_key = config.app.get("openai_api_key")
                model_name = config.app.get("openai_model_name")
                base_url = config.app.get("openai_base_url", "")
                if not base_url:
                    base_url = "https://api.openai.com/v1"
            elif llm_provider == "aihubmix":
                api_key = config.app.get("aihubmix_api_key")
                model_name = config.app.get("aihubmix_model_name")
                base_url = config.app.get("aihubmix_base_url", "")
                # AIHubMix 兼容 OpenAI Chat Completions 协议。这里使用独立
                # provider 保存合作方的默认网关和推荐模型，避免把推广链接、
                # 默认模型等合作配置混进普通 OpenAI provider，影响现有用户。
                if not base_url:
                    base_url = "https://aihubmix.com/v1"
                if not model_name:
                    model_name = "gpt-5.4-mini"
            elif llm_provider == "oneapi":
                api_key = config.app.get("oneapi_api_key")
                model_name = config.app.get("oneapi_model_name")
                base_url = config.app.get("oneapi_base_url", "")
            elif llm_provider == "azure":
                api_key = config.app.get("azure_api_key")
                model_name = config.app.get("azure_model_name")
                base_url = config.app.get("azure_base_url", "")
                api_version = config.app.get("azure_api_version", "2024-02-15-preview")
            elif llm_provider == "gemini":
                api_key = config.app.get("gemini_api_key")
                model_name = config.app.get("gemini_model_name")
                base_url = config.app.get("gemini_base_url", "")
                # Gemini 旧模型名已经陆续下线，这里自动兼容历史配置，
                # 避免用户沿用旧值时直接收到 404。
                if not model_name:
                    model_name = _DEFAULT_GEMINI_MODEL
                elif model_name in _DEPRECATED_GEMINI_MODELS:
                    logger.warning(
                        f"gemini model '{model_name}' is deprecated, fallback to '{_DEFAULT_GEMINI_MODEL}'"
                    )
                    model_name = _DEFAULT_GEMINI_MODEL
            elif llm_provider == "grok":
                api_key = config.app.get("grok_api_key")
                model_name = config.app.get("grok_model_name")
                base_url = config.app.get("grok_base_url", "")
                if not base_url:
                    base_url = "https://api.x.ai/v1"
            elif llm_provider == "groq":
                api_key = config.app.get("groq_api_key")
                model_name = config.app.get("groq_model_name")
                if not model_name:
                    model_name = "llama-3.3-70b-versatile"
                base_url = config.app.get("groq_base_url", "")
                if not base_url:
                    base_url = "https://api.groq.com/openai/v1"
            elif llm_provider == "qwen":
                api_key = config.app.get("qwen_api_key")
                model_name = config.app.get("qwen_model_name")
                base_url = "***"
            elif llm_provider == "cloudflare":
                api_key = config.app.get("cloudflare_api_key")
                model_name = config.app.get("cloudflare_model_name")
                account_id = config.app.get("cloudflare_account_id")
                base_url = "***"
            elif llm_provider == "minimax":
                api_key = config.app.get("minimax_api_key")
                model_name = config.app.get("minimax_model_name")
                base_url = config.app.get("minimax_base_url", "")
                if not base_url:
                    base_url = "https://api.minimax.io/v1"
            elif llm_provider == "mimo":
                api_key = config.app.get("mimo_api_key")
                model_name = config.app.get("mimo_model_name")
                base_url = config.app.get("mimo_base_url", "")
                # Xiaomi MiMo 官方文档说明其兼容 OpenAI Chat Completions 协议。
                # 这里使用独立 provider 保存默认地址和模型名，用户不用把 MiMo
                # 当作 OpenAI 自定义 base_url 配置，也便于后续继续接入 MiMo
                # 多模态或 TTS 能力时保持边界清晰。
                if not base_url:
                    base_url = "https://api.xiaomimimo.com/v1"
                if not model_name:
                    model_name = "mimo-v2.5-pro"
            elif llm_provider == "deepseek":
                api_key = config.app.get("deepseek_api_key")
                model_name = config.app.get("deepseek_model_name")
                base_url = config.app.get("deepseek_base_url")
                if not base_url:
                    base_url = "https://api.deepseek.com"
            elif llm_provider == "modelscope":
                api_key = config.app.get("modelscope_api_key")
                model_name = config.app.get("modelscope_model_name")
                base_url = config.app.get("modelscope_base_url")
                if not base_url:
                    base_url = "https://api-inference.modelscope.cn/v1/"
            elif llm_provider == "ernie":
                api_key = config.app.get("ernie_api_key")
                secret_key = config.app.get("ernie_secret_key")
                base_url = config.app.get("ernie_base_url")
                model_name = "***"
                if not secret_key:
                    raise ValueError(
                        f"{llm_provider}: secret_key is not set, please set it in the config.toml file."
                    )
            elif llm_provider == "pollinations":
                try:
                    base_url = config.app.get("pollinations_base_url", "")
                    if not base_url:
                        base_url = "https://text.pollinations.ai/openai"
                    model_name = config.app.get("pollinations_model_name", "openai-fast")
                   
                    # Prepare the payload
                    payload = {
                        "model": model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "seed": 101  # Optional but helps with reproducibility
                    }
                    
                    # Optional parameters if configured
                    if config.app.get("pollinations_private"):
                        payload["private"] = True
                    if config.app.get("pollinations_referrer"):
                        payload["referrer"] = config.app.get("pollinations_referrer")
                    
                    headers = {
                        "Content-Type": "application/json"
                    }
                    
                    # Make the API request
                    response = requests.post(base_url, headers=headers, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    
                    if result and "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        return _normalize_text_response(content, llm_provider)
                    else:
                        raise Exception(f"[{llm_provider}] returned an invalid response format")
                        
                except requests.exceptions.RequestException as e:
                    raise Exception(f"[{llm_provider}] request failed: {str(e)}")
                except Exception as e:
                    raise Exception(f"[{llm_provider}] error: {str(e)}")

            elif llm_provider == "litellm":
                model_name = config.app.get("litellm_model_name")

            if llm_provider not in ["pollinations", "ollama", "litellm"]:  # Skip validation for providers that don't require API key
                if not api_key:
                    raise ValueError(
                        f"{llm_provider}: api_key is not set, please set it in the config.toml file."
                    )
                if not model_name:
                    raise ValueError(
                        f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                    )
                if not base_url and llm_provider not in ["gemini"]:
                    raise ValueError(
                        f"{llm_provider}: base_url is not set, please set it in the config.toml file."
                    )

            if llm_provider == "qwen":
                import dashscope
                from dashscope.api_entities.dashscope_response import GenerationResponse

                dashscope.api_key = api_key
                response = dashscope.Generation.call(
                    model=model_name, messages=[{"role": "user", "content": prompt}]
                )
                if response:
                    if isinstance(response, GenerationResponse):
                        status_code = response.status_code
                        if status_code != 200:
                            raise Exception(
                                f'[{llm_provider}] returned an error response: "{response}"'
                            )

                        return _extract_qwen_generation_text(response)
                    else:
                        raise Exception(
                            f'[{llm_provider}] returned an invalid response: "{response}"'
                        )
                else:
                    raise Exception(f"[{llm_provider}] returned an empty response")

            if llm_provider == "gemini":
                import google.generativeai as genai

                if not base_url:
                    genai.configure(api_key=api_key, transport="rest")
                else:
                    genai.configure(api_key=api_key, transport="rest", client_options={'api_endpoint': base_url})

                generation_config = {
                    "temperature": 0.5,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 2048,
                }

                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                ]

                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )

                try:
                    response = model.generate_content(prompt)
                    candidates = response.candidates
                    generated_text = candidates[0].content.parts[0].text
                except (AttributeError, IndexError) as e:
                    logger.warning(
                        f"gemini returned invalid response content: {str(e)}"
                    )
                    raise ValueError(
                        f"[{llm_provider}] returned invalid response content"
                    )

                return _normalize_text_response(generated_text, llm_provider)

            if llm_provider == "cloudflare":
                response = requests.post(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a friendly assistant",
                            },
                            {"role": "user", "content": prompt},
                        ]
                    },
                )
                result = response.json()
                logger.info(result)
                return _normalize_text_response(result["result"]["response"], llm_provider)

            if llm_provider == "ernie":
                response = requests.post(
                    "https://aip.baidubce.com/oauth/2.0/token", 
                    params={
                        "grant_type": "client_credentials",
                        "client_id": api_key,
                        "client_secret": secret_key,
                    }
                )
                access_token = response.json().get("access_token")
                url = f"{base_url}?access_token={access_token}"

                payload = json.dumps(
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.5,
                        "top_p": 0.8,
                        "penalty_score": 1,
                        "disable_search": False,
                        "enable_citation": False,
                        "response_format": "text",
                    }
                )
                headers = {"Content-Type": "application/json"}

                response = requests.request(
                    "POST", url, headers=headers, data=payload
                ).json()
                return _normalize_text_response(response.get("result"), llm_provider)

            if llm_provider == "litellm":
                import litellm

                if not model_name:
                    raise ValueError(
                        f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                    )

                response = litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    drop_params=True,
                )

                if not response:
                    raise ValueError(f"[{llm_provider}] returned empty response")
                if not getattr(response, "choices", None):
                    raise ValueError(f"[{llm_provider}] returned empty response")

                return _extract_chat_completion_text(response, llm_provider)

            if llm_provider == "azure":
                # Azure OpenAI SDK 使用 `azure_endpoint` 和 `api_version` 生成专用请求地址，
                # 不能继续复用下面普通 OpenAI-compatible 的 `base_url` 初始化逻辑。
                # 这里在 Azure 分支内完成请求并立即返回，避免客户端被后续 fallback
                # 覆盖，导致用户配置的 Azure 凭证通过校验但实际请求没有被使用。
                logger.info(f"requesting azure chat completion, model: {model_name}")
                client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=base_url,
                )
                response = client.chat.completions.create(
                    model=model_name, messages=[{"role": "user", "content": prompt}]
                )
                if response:
                    if isinstance(response, ChatCompletion):
                        return _extract_chat_completion_text(response, llm_provider)
                    else:
                        raise Exception(
                            f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                            f"connection and try again."
                        )
                else:
                    raise Exception(
                        f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                    )

            if llm_provider == "modelscope":
                content = ''
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"enable_thinking": False},
                    stream=True
                )
                if response:
                    for chunk in response:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            content += delta.content
                    
                    if not content.strip():
                        raise ValueError("Empty content in stream response")
                    
                    return _normalize_text_response(content, llm_provider)
                else:
                    raise Exception(f"[{llm_provider}] returned an empty response")

            else:
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )

            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    return _extract_chat_completion_text(response, llm_provider)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                        f"connection and try again."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                )

        return _normalize_text_response(content, llm_provider)
    except Exception as e:
        return f"Error: {str(e)}"


def _limit_script_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API 层已经用 Pydantic 做长度校验；这里继续兜底，是为了保护
    # WebUI 或内部服务直接调用 generate_script 时不会把超长提示词发送给模型，
    # 避免 token 成本异常和请求失败。
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _normalize_script_paragraph_number(paragraph_number: int | None) -> int:
    try:
        value = int(paragraph_number or MIN_SCRIPT_PARAGRAPH_NUMBER)
    except (TypeError, ValueError):
        value = MIN_SCRIPT_PARAGRAPH_NUMBER

    if value < MIN_SCRIPT_PARAGRAPH_NUMBER or value > MAX_SCRIPT_PARAGRAPH_NUMBER:
        # WebUI 和 API 都会限制范围；这里兜底处理内部调用，避免异常参数直接扩大
        # LLM 生成成本或生成空结果。
        logger.warning(
            "script paragraph_number is out of range and will be clamped: "
            f"{value}"
        )
        return max(MIN_SCRIPT_PARAGRAPH_NUMBER, min(value, MAX_SCRIPT_PARAGRAPH_NUMBER))

    return value


def build_script_prompt(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )

    # 将“脚本生成规则”和“运行时上下文”分开拼接。这样高级用户即使覆盖默认
    # system prompt，也不会漏掉视频主题、语言、段落数这些每次生成都必须带上的参数。
    prompt = custom_system_prompt or DEFAULT_SCRIPT_SYSTEM_PROMPT
    prompt += f"""

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".rstrip()
    if language:
        prompt += f"\n- language: {language}"
    if video_script_prompt:
        prompt += f"""

# Additional User Requirements:
{video_script_prompt}
""".rstrip()

    return prompt


def generate_script(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )
    prompt = build_script_prompt(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
    )
    final_script = ""
    logger.info(
        "generating video script: "
        f"subject={video_subject}, paragraph_number={paragraph_number}, "
        f"has_custom_prompt={bool(video_script_prompt.strip())}, "
        f"has_custom_system_prompt={bool(custom_system_prompt.strip())}"
    )

    def format_response(response):
        # Clean the script
        # Remove asterisks, hashes
        response = response.replace("*", "")
        response = response.replace("#", "")

        # Remove markdown syntax
        response = re.sub(r"\[.*\]", "", response)
        response = re.sub(r"\(.*\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        # selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        return "\n\n".join(paragraphs)

    for i in range(_max_retries):
        try:
            response = _generate_response(prompt=prompt)
            if response:
                final_script = format_response(response)
            else:
                logging.error("gpt returned an empty response")

            # g4f may return an error message
            if final_script and "当日额度已消耗完" in final_script:
                raise ValueError(final_script)

            if final_script:
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")

        if i < _max_retries:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")
    if "Error: " in final_script:
        logger.error(f"failed to generate video script: {final_script}")
    else:
        logger.success(f"completed: \n{final_script}")
    return final_script.strip()


def generate_terms(video_subject: str, video_script: str, amount: int = 5) -> List[str]:
    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
Generate {amount} search terms for stock videos, depending on the subject of a video.

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words, always add the main subject of the video.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video.
5. reply with english search terms only.

## Output Example:
["search term 1", "search term 2", "search term 3","search term 4","search term 5"]

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if "Error: " in response:
                logger.error(f"failed to generate video script: {response}")
                return response
            search_terms = json.loads(response)
            if not isinstance(search_terms, list) or not all(
                isinstance(term, str) for term in search_terms
            ):
                logger.error("response is not a list of strings.")
                continue

        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                match = re.search(r"\[.*]", response)
                if match:
                    try:
                        search_terms = json.loads(match.group())
                    except Exception as e:
                        # 这里保留重试流程，但必须记录 LLM 返回的非标准 JSON，
                        # 否则后续排查搜索词为空时无法定位
                        # 是模型格式问题还是解析逻辑问题。
                        logger.warning(f"failed to generate video terms: {str(e)}")

        if search_terms and len(search_terms) > 0:
            break
        if i < _max_retries:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    logger.success(f"completed: \n{search_terms}")
    return search_terms


# =============================================================================
# Social publishing metadata
#
# 根据视频主题和脚本生成发布到短视频平台时常用的 title、caption 和 hashtags。
# 这块能力只复用现有 LLM provider，不接入任何外部发布服务，也不影响视频生成主链路。
# =============================================================================

# 不同平台的文案长度和 hashtag 数量偏好不同。这里使用保守上限，避免模型返回
# 过长内容后调用方还需要二次裁剪。
SOCIAL_PLATFORMS = {
    "tiktok": {"title_max": 100, "caption_max": 2200, "hashtag_count": 5},
    "youtube_shorts": {"title_max": 100, "caption_max": 5000, "hashtag_count": 3},
    "instagram_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 8},
    "facebook_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 5},
}
DEFAULT_SOCIAL_PLATFORM = "tiktok"
DEFAULT_SOCIAL_LANGUAGE = "auto"
MAX_SOCIAL_SUBJECT_LENGTH = 500
MAX_SOCIAL_SCRIPT_LENGTH = 8000
MAX_SOCIAL_LANGUAGE_LENGTH = 64

SOCIAL_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube_shorts": "YouTube Shorts",
    "instagram_reels": "Instagram Reels",
    "facebook_reels": "Facebook Reels",
}

# LLM 不可用时的通用兜底标签。这里故意不绑定某个国家或语种，保证 API
# 对中文、英文、越南语等不同场景都能返回可用结构。
DEFAULT_SOCIAL_HASHTAGS = [
    "#shorts",
    "#viral",
    "#trending",
    "#fyp",
    "#video",
    "#reels",
    "#creator",
    "#content",
]


def _resolve_social_platform(platform: str | None) -> str:
    value = (platform or "").strip().lower()
    return value if value in SOCIAL_PLATFORMS else DEFAULT_SOCIAL_PLATFORM


def _normalize_social_language(language: str | None) -> str:
    value = (language or DEFAULT_SOCIAL_LANGUAGE).strip()
    if len(value) > MAX_SOCIAL_LANGUAGE_LENGTH:
        logger.warning(
            "social metadata language is too long and will be truncated to "
            f"{MAX_SOCIAL_LANGUAGE_LENGTH} characters."
        )
        value = value[:MAX_SOCIAL_LANGUAGE_LENGTH]
    return value or DEFAULT_SOCIAL_LANGUAGE


def _limit_social_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # API 层会限制长度；这里继续兜底，是为了保护内部调用或未来 WebUI
    # 直接调用时不会把超长内容发送给模型，避免 token 成本异常。
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _social_language_instruction(language: str | None) -> str:
    language = _normalize_social_language(language)
    if language.lower() == DEFAULT_SOCIAL_LANGUAGE:
        return (
            "Use the same language as the video subject and script. If the subject "
            "and script use different languages, prefer the script language."
        )

    return f'Write "title" and "caption" in this language: {language}.'


def _clamp_text(text, max_length: int) -> str:
    value = ("" if text is None else str(text)).strip()
    if max_length and len(value) > max_length:
        return value[:max_length].rstrip()
    return value


def _normalize_hashtags(raw, count: int) -> List[str]:
    """
    将 LLM 返回的 hashtag 统一整理成 `#tag` 格式。

    LLM 可能返回字符串、数组、带空格的词组、重复标签或包含标点的内容。
    这里集中清洗，可以让接口响应结构稳定，也避免平台发布时出现空标签、
    重复标签或不符合常见格式的 hashtag。
    """
    if isinstance(raw, str):
        candidates = re.split(r"[\s,]+", raw)
    elif isinstance(raw, (list, tuple)):
        # 数组里的每一项视为一个完整标签，因此 "du lich" 会变成
        # "#dulich"，而不是拆成两个标签。
        candidates = [str(entry) for entry in raw]
    else:
        candidates = []

    seen = set()
    result: List[str] = []
    for item in candidates:
        tag = re.sub(r"[^\w]", "", item, flags=re.UNICODE)
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(f"#{tag}")
        if count and len(result) >= count:
            break
    return result


def build_social_metadata_prompt(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> str:
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    platform = _resolve_social_platform(platform)
    spec = SOCIAL_PLATFORMS[platform]
    label = SOCIAL_PLATFORM_LABELS.get(platform, platform)
    language_instruction = _social_language_instruction(language)

    prompt = f"""
# Role: Short-Video Social Media Copywriter

## Goal
Write engaging publishing metadata for a short video that will be posted on {label}.

## Constraints
1. Respond ONLY with a single valid minified JSON object. No markdown, no code fences, no commentary.
2. The JSON must contain exactly these keys: "title", "caption", "hashtags".
3. "title": a catchy hook, at most {spec['title_max']} characters.
4. "caption": an engaging description that ends with a call to action, at most {spec['caption_max']} characters. Do not put hashtags inside the caption.
5. "hashtags": a JSON array of exactly {spec['hashtag_count']} strings. Each must start with "#", contain no spaces, and be relevant to the topic and to {label}.
6. {language_instruction}

## Output Example
{{"title":"...","caption":"...","hashtags":["#example","#video"]}}

## Context
### Video Subject
{video_subject}

### Video Script
{video_script}
""".strip()
    return prompt


def _parse_social_metadata(response: str, platform: str) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]

    data = None
    try:
        data = json.loads(response)
    except Exception:
        # 部分模型会在 JSON 外层包一段说明文字或 markdown fence。
        # API 调用方只需要稳定结构，所以这里尝试提取第一个 JSON object。
        match = re.search(r"\{.*\}", response or "", re.DOTALL)
        if match:
            data = json.loads(match.group())

    if not isinstance(data, dict):
        raise ValueError("social metadata response is not a JSON object")

    title = _clamp_text(data.get("title", ""), spec["title_max"])
    caption = _clamp_text(data.get("caption", ""), spec["caption_max"])
    hashtags = _normalize_hashtags(data.get("hashtags", []), spec["hashtag_count"])

    if not title and not caption:
        raise ValueError("social metadata response is missing both title and caption")

    return {"title": title, "caption": caption, "hashtags": hashtags}


def _fallback_social_metadata(
    video_subject: str, video_script: str, platform: str
) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]
    subject = (video_subject or "").strip()
    script = (video_script or "").strip()

    title = subject
    if not title and script:
        # 没有主题时，用脚本第一句兜底生成 title，避免接口返回空标题。
        title = re.split(r"(?<=[.!?。！？])\s+", script)[0]

    return {
        "title": _clamp_text(title, spec["title_max"]),
        "caption": _clamp_text(script or subject, spec["caption_max"]),
        "hashtags": _normalize_hashtags(
            DEFAULT_SOCIAL_HASHTAGS, spec["hashtag_count"]
        ),
    }


def generate_social_metadata(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> dict:
    """
    生成短视频发布文案元数据。

    返回结构固定为 `{"title": str, "caption": str, "hashtags": List[str]}`。
    如果 LLM 不可用或返回格式异常，会降级为通用启发式结果，保证 API
    调用方始终拿到可展示、可发布前编辑的数据结构。
    """
    platform = _resolve_social_platform(platform)
    language = _normalize_social_language(language)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    prompt = build_social_metadata_prompt(
        video_subject=video_subject,
        video_script=video_script,
        language=language,
        platform=platform,
    )
    logger.info(
        f"generating social metadata: platform={platform}, language={language}"
    )

    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and "Error: " in response:
                logger.error(f"failed to generate social metadata: {response}")
                break
            metadata = _parse_social_metadata(response, platform)
            logger.success(f"completed: \n{metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"failed to parse social metadata: {str(e)}")

        if i < _max_retries - 1:
            logger.warning(
                f"failed to generate social metadata, trying again... {i + 1}"
            )

    logger.warning("falling back to heuristic social metadata")
    return _fallback_social_metadata(video_subject, video_script, platform)


# =============================================================================
# TikTok affiliate product idea finder
#
# 用现有 LLM provider 给出"适合做 TikTok 带货短视频"的选题建议。注意：这是
# 模型基于通用电商趋势的启发式建议，并非实时真实销量数据。不接入任何外部
# 数据源，也不影响视频生成主链路。
# =============================================================================

DEFAULT_PRODUCT_IDEA_COUNT = 6
MAX_PRODUCT_IDEA_COUNT = 12
MAX_PRODUCT_FIELD_LENGTH = 200
PRODUCT_IDEA_KEYS = ("product", "category", "reason", "audience", "angle")


def _normalize_product_idea_count(amount) -> int:
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return DEFAULT_PRODUCT_IDEA_COUNT
    return max(1, min(MAX_PRODUCT_IDEA_COUNT, amount))


def _coerce_product_idea(item) -> dict:
    """Normalize one LLM-returned idea into the fixed {key: str} shape, dropping
    items that are not dicts or have no product name."""
    if not isinstance(item, dict):
        return {}
    idea = {}
    for key in PRODUCT_IDEA_KEYS:
        value = item.get(key, "")
        idea[key] = _clamp_text(value, MAX_PRODUCT_FIELD_LENGTH)
    if not idea["product"]:
        return {}
    return idea


def _generate_json_object_list(prompt: str, coerce, label: str) -> List[dict]:
    """Shared retry loop for the affiliate generators that expect a JSON array of
    objects back from the LLM.

    Calls the model up to ``_max_retries`` times, parsing the response as a JSON
    array (recovering an array embedded in surrounding prose when needed), maps
    each element through ``coerce`` and drops falsy results. ``label`` is only
    used in log messages. Returns the cleaned list (NOT truncated — the caller
    applies its own ``[:amount]``). Returns [] if the model reports an
    ``Error: ...`` or every attempt fails to yield items.
    """
    items: List[dict] = []
    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and "Error: " in response:
                logger.error(f"failed to generate {label}: {response}")
                break
            raw = json.loads(response)
        except Exception as e:
            logger.warning(f"failed to parse {label}: {str(e)}")
            raw = None
            if response:
                match = re.search(r"\[.*]", response, re.DOTALL)
                if match:
                    try:
                        raw = json.loads(match.group())
                    except Exception as e:
                        logger.warning(f"failed to parse {label} json: {str(e)}")

        if isinstance(raw, list):
            items = [item for item in (coerce(x) for x in raw) if item]

        if items:
            break
        if i < _max_retries - 1:
            logger.warning(f"failed to generate {label}, trying again... {i + 1}")

    logger.success(f"completed: {len(items)} {label}")
    return items


def generate_product_ideas(
    category: str = "",
    market: str = "",
    language: str = "",
    amount: int = DEFAULT_PRODUCT_IDEA_COUNT,
) -> List[dict]:
    """Suggest TikTok-affiliate product/niche ideas.

    Returns a list of dicts with the keys in PRODUCT_IDEA_KEYS. These are
    AI-generated suggestions from general short-video commerce trends, NOT
    real-time sales data. On repeated failure an empty list is returned so the
    caller can show a friendly message rather than crash.
    """
    amount = _normalize_product_idea_count(amount)
    category = _limit_social_text(category, MAX_SOCIAL_SUBJECT_LENGTH, "category")
    market = _limit_social_text(market, MAX_SOCIAL_SUBJECT_LENGTH, "market")
    language = (language or "").strip()

    category_line = (
        f"Focus on this product category / niche: {category}."
        if category
        else "Cover a diverse mix of categories that are popular for TikTok affiliate selling."
    )
    market_line = f"Target market / region: {market}." if market else ""
    language_line = (
        f'Write every text field ("product", "category", "reason", "audience", "angle") in this language: {language}.'
        if language
        else "Write every text field in clear, simple language matching the category."
    )

    prompt = f"""
# Role: TikTok Affiliate Product Idea Generator

## Goals:
Suggest {amount} product ideas that tend to sell well for TikTok affiliate marketing and are easy to turn into a short promo video.

## Important:
1. these are AI suggestions based on general TikTok / short-video commerce trends, not real-time sales numbers.
2. prefer affordable, visually demonstrable, impulse-buy friendly products that perform well in short videos.
3. avoid restricted or risky categories (weapons, drugs, adult products, medical or financial guarantees, counterfeits).

## Constrains:
1. return ONLY a json-array of objects. do not return any text before or after the json.
2. each object must have exactly these keys: "product", "category", "reason", "audience", "angle".
   - "product": a specific product type, short.
   - "category": the niche or category it belongs to.
   - "reason": one short sentence on why it sells well on TikTok.
   - "audience": who typically buys it.
   - "angle": a short video hook or content angle idea.
3. {language_line}
4. {category_line}
{("5. " + market_line) if market_line else ""}

## Output Example:
[{{"product": "...", "category": "...", "reason": "...", "audience": "...", "angle": "..."}}]
""".strip()

    logger.info(
        f"generating product ideas: category={category!r}, market={market!r}, amount={amount}"
    )

    ideas = _generate_json_object_list(prompt, _coerce_product_idea, "product ideas")
    return ideas[:amount]


# =============================================================================
# TikTok affiliate hook generator
#
# 为一个视频主题生成多条"前 3 秒"的开场钩子（hook）。带货短视频的完播率几乎
# 完全取决于开头是否抓人，所以让用户一次拿到多条可 A/B 测试的开场白很有价值。
# 复用现有 LLM provider，不接入外部数据，也不影响视频生成主链路。
# =============================================================================

DEFAULT_HOOK_COUNT = 5
MAX_HOOK_COUNT = 10
MAX_HOOK_LENGTH = 200


def _normalize_hook_count(amount) -> int:
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return DEFAULT_HOOK_COUNT
    return max(1, min(MAX_HOOK_COUNT, amount))


def _coerce_hook(item) -> str:
    """Normalize one LLM-returned hook into a clean single-line string, stripping
    leading list markers / numbering / surrounding quotes the model often adds."""
    if not isinstance(item, str):
        return ""
    hook = item.strip()
    # Drop a leading "1." / "1)" / "-" / "*" / "•" bullet or numbering.
    hook = re.sub(r'^\s*(?:\d+[.)]|[-*•])\s*', "", hook)
    # Drop matching wrapping quotes.
    if len(hook) >= 2 and hook[0] in "\"'“”" and hook[-1] in "\"'“”":
        hook = hook[1:-1].strip()
    return _clamp_text(hook, MAX_HOOK_LENGTH)


def generate_hook_variations(
    video_subject: str,
    language: str = "",
    amount: int = DEFAULT_HOOK_COUNT,
) -> List[str]:
    """Generate several scroll-stopping opening hook lines for a TikTok affiliate
    video about ``video_subject``.

    A hook is the first spoken line (first ~3 seconds) that stops the scroll.
    Returns a de-duplicated list of plain-text hook strings. On repeated failure
    an empty list is returned so the caller can show a friendly message rather
    than crash.
    """
    amount = _normalize_hook_count(amount)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    language = (language or "").strip()

    language_line = (
        f"Write every hook in this language: {language}."
        if language
        else "Write every hook in the same language as the video subject."
    )

    prompt = f"""
# Role: TikTok Affiliate Hook Writer

## Goals:
Write {amount} different scroll-stopping opening lines (hooks) for a short TikTok affiliate video about this subject: "{video_subject}".

## What makes a good hook:
1. it is the very first thing said, designed to stop the scroll within 3 seconds.
2. use proven angles: a bold claim, a relatable problem, curiosity, a surprising result, or a "stop doing X" pattern interrupt.
3. keep each hook to one short spoken sentence that is easy to read aloud and fits on a subtitle.

## Constrains:
1. return ONLY a json-array of strings. do not return any text before or after the json.
2. each array element is one hook, as plain spoken text.
3. do not include numbering, quotes, markdown, emojis, hashtags, or stage directions.
4. do not invent fake prices, fake discounts, or unverifiable statistics.
5. make the {amount} hooks clearly different from each other in angle and wording.
6. {language_line}

## Output Example:
["Stop scrolling if your ... keeps ...", "I wish someone told me this sooner ...", "..."]
""".strip()

    logger.info(
        f"generating hooks: subject={video_subject!r}, amount={amount}, language={language!r}"
    )

    hooks: List[str] = []
    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and "Error: " in response:
                logger.error(f"failed to generate hooks: {response}")
                break
            raw = json.loads(response)
        except Exception as e:
            logger.warning(f"failed to parse hooks: {str(e)}")
            raw = None
            if response:
                match = re.search(r"\[.*]", response, re.DOTALL)
                if match:
                    try:
                        raw = json.loads(match.group())
                    except Exception as e:
                        logger.warning(f"failed to parse hooks json: {str(e)}")

        if isinstance(raw, list):
            seen = set()
            for x in raw:
                hook = _coerce_hook(x)
                key = hook.lower()
                if hook and key not in seen:
                    seen.add(key)
                    hooks.append(hook)

        if hooks:
            break
        if i < _max_retries - 1:
            logger.warning(f"failed to generate hooks, trying again... {i + 1}")

    logger.success(f"completed: {len(hooks)} hooks")
    return hooks[:amount]


# =============================================================================
# TikTok affiliate shot list / storyboard generator
#
# 把已经写好的口播脚本拆成一条条「分镜」，每个镜头给出：要拍/要展示的画面、
# 这一句的口播文案、屏幕上的字幕贴纸、以及一个可直接拿去搜素材的 b-roll 关键词。
# 带货创作者最常卡在「文案有了但不知道怎么拍」，这个功能把脚本变成可执行的拍摄
# 清单。复用现有 LLM provider，不接外部数据，也不影响视频生成主链路。
# =============================================================================

DEFAULT_SHOT_COUNT = 6
MAX_SHOT_COUNT = 15
MAX_SHOT_FIELD_LENGTH = 200
SHOT_KEYS = ("scene", "voiceover", "onscreen_text", "broll")


def _normalize_shot_count(amount) -> int:
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return DEFAULT_SHOT_COUNT
    return max(1, min(MAX_SHOT_COUNT, amount))


def _coerce_shot(item) -> dict:
    """Normalize one LLM-returned shot into the fixed {key: str} shape, dropping
    items that are not dicts or describe no scene to film."""
    if not isinstance(item, dict):
        return {}
    shot = {}
    for key in SHOT_KEYS:
        value = item.get(key, "")
        shot[key] = _clamp_text(value, MAX_SHOT_FIELD_LENGTH)
    if not shot["scene"]:
        return {}
    return shot


def generate_shot_list(
    video_subject: str,
    video_script: str = "",
    language: str = "",
    amount: int = DEFAULT_SHOT_COUNT,
) -> List[dict]:
    """Break an affiliate video script into a shot-by-shot shooting plan.

    Returns a list of dicts with the keys in SHOT_KEYS (scene, voiceover,
    onscreen_text, broll). The shots follow the order of the script so the
    creator can film them top to bottom. On repeated failure an empty list is
    returned so the caller can show a friendly message rather than crash.
    """
    amount = _normalize_shot_count(amount)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_script_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    language = (language or "").strip()

    script_line = (
        f'Base the shots on this existing script:\n"""\n{video_script}\n"""'
        if video_script
        else "No script was provided, so plan shots that would naturally tell this product's story."
    )
    language_line = (
        f'Write the "scene", "voiceover", "onscreen_text" and "broll" fields in this language: {language}.'
        if language
        else "Write every text field in the same language as the subject / script."
    )

    prompt = f"""
# Role: TikTok Affiliate Video Director

## Goals:
Turn a short affiliate video about "{video_subject}" into a clear shot-by-shot shooting plan of about {amount} shots.

## Context:
{script_line}

## What each shot needs:
1. follow the natural order of the video (hook first, then benefits / demo, then call-to-action last).
2. be simple and realistic to film on a phone by a solo creator.

## Constrains:
1. return ONLY a json-array of objects. do not return any text before or after the json.
2. each object must have exactly these keys: "scene", "voiceover", "onscreen_text", "broll".
   - "scene": what to film or show on screen for this shot, short and concrete.
   - "voiceover": the spoken line for this shot (reuse / lightly trim the script if one is given).
   - "onscreen_text": a very short text sticker / caption to overlay (a few words).
   - "broll": one short stock-footage search keyword if real filming is hard, else "".
3. {language_line}
4. produce about {amount} shots, ordered from first to last.
5. do not invent fake prices, fake discounts, or unverifiable statistics.

## Output Example:
[{{"scene": "...", "voiceover": "...", "onscreen_text": "...", "broll": "..."}}]
""".strip()

    logger.info(
        f"generating shot list: subject={video_subject!r}, amount={amount}, "
        f"has_script={bool(video_script)}, language={language!r}"
    )

    shots = _generate_json_object_list(prompt, _coerce_shot, "shots")
    return shots[:amount]


# =============================================================================
# TikTok affiliate comment-reply / objection-handling generator
#
# 带货短视频的转化很大一部分发生在评论区：观众会问「多少钱」「在哪买」「真的有用
# 吗」。这个功能预测最常见的提问/异议，并给出可直接复制粘贴、能把人引导到购物车/
# 主页链接的回复，帮创作者更快地把评论变成订单。复用现有 LLM provider，不接外部
# 数据，也不影响视频生成主链路。
# =============================================================================

DEFAULT_COMMENT_REPLY_COUNT = 6
MAX_COMMENT_REPLY_COUNT = 12
MAX_COMMENT_REPLY_FIELD_LENGTH = 300
COMMENT_REPLY_KEYS = ("comment", "reply")


def _normalize_comment_reply_count(amount) -> int:
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return DEFAULT_COMMENT_REPLY_COUNT
    return max(1, min(MAX_COMMENT_REPLY_COUNT, amount))


def _coerce_comment_reply(item) -> dict:
    """Normalize one LLM-returned pair into the fixed {comment, reply} shape,
    dropping items that are not dicts or are missing either side."""
    if not isinstance(item, dict):
        return {}
    pair = {}
    for key in COMMENT_REPLY_KEYS:
        value = item.get(key, "")
        pair[key] = _clamp_text(value, MAX_COMMENT_REPLY_FIELD_LENGTH)
    if not pair["comment"] or not pair["reply"]:
        return {}
    return pair


def generate_comment_replies(
    video_subject: str,
    language: str = "",
    amount: int = DEFAULT_COMMENT_REPLY_COUNT,
) -> List[dict]:
    """Predict the most common viewer comments / objections on an affiliate video
    and draft a ready-to-paste reply for each that nudges toward the link.

    Returns a list of dicts with the keys in COMMENT_REPLY_KEYS (comment, reply).
    On repeated failure an empty list is returned so the caller can show a
    friendly message rather than crash.
    """
    amount = _normalize_comment_reply_count(amount)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    language = (language or "").strip()

    language_line = (
        f'Write the "comment" and "reply" fields in this language: {language}.'
        if language
        else "Write every text field in the same language as the subject."
    )

    prompt = f"""
# Role: TikTok Affiliate Community Manager

## Goals:
Predict {amount} of the most common comments viewers leave on a short affiliate video about "{video_subject}", and write a short reply for each that politely answers and nudges the viewer toward the product link.

## Cover a realistic mix:
1. buying questions ("how much is it", "where do I get it", "is there a link").
2. doubts / objections ("does it actually work", "looks cheap", "too expensive").
3. positive comments worth converting ("I need this", "omg").

## Constrains:
1. return ONLY a json-array of objects. do not return any text before or after the json.
2. each object must have exactly these keys: "comment", "reply".
   - "comment": a short, realistic viewer comment.
   - "reply": a friendly 1-2 sentence reply that answers and points to the link in bio / cart.
3. {language_line}
4. do not invent fake prices, fake discounts, fake reviews, or unverifiable claims.
5. keep replies honest and non-spammy; no guarantees about results or health/financial outcomes.

## Output Example:
[{{"comment": "where can I buy this?", "reply": "..."}}]
""".strip()

    logger.info(
        f"generating comment replies: subject={video_subject!r}, amount={amount}, "
        f"language={language!r}"
    )

    replies = _generate_json_object_list(
        prompt, _coerce_comment_reply, "comment replies"
    )
    return replies[:amount]


# =============================================================================
# TikTok affiliate trending-sound idea generator
#
# 配乐对带货短视频的播放量影响很大，但创作者常常不知道该配什么风格的声音。这个
# 功能根据视频主题给出「适合的声音风格 + 可以去声音库搜索的关键词 + 用法建议」。
# 注意：模型无法访问实时热榜，所以这是基于通用短视频配乐规律的建议，而不是当下的
# 实时热门曲目。复用现有 LLM provider，不接外部数据，也不影响视频生成主链路。
# =============================================================================

DEFAULT_SOUND_COUNT = 5
MAX_SOUND_COUNT = 10
MAX_SOUND_FIELD_LENGTH = 200
SOUND_KEYS = ("sound", "vibe", "search", "tip")


def _normalize_sound_count(amount) -> int:
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return DEFAULT_SOUND_COUNT
    return max(1, min(MAX_SOUND_COUNT, amount))


def _coerce_sound_idea(item) -> dict:
    """Normalize one LLM-returned sound idea into the fixed {key: str} shape,
    dropping items that are not dicts or describe no sound."""
    if not isinstance(item, dict):
        return {}
    idea = {}
    for key in SOUND_KEYS:
        value = item.get(key, "")
        idea[key] = _clamp_text(value, MAX_SOUND_FIELD_LENGTH)
    if not idea["sound"]:
        return {}
    return idea


def generate_sound_ideas(
    video_subject: str,
    language: str = "",
    amount: int = DEFAULT_SOUND_COUNT,
) -> List[dict]:
    """Suggest sound / music styles that tend to fit an affiliate video about
    ``video_subject``, plus a keyword to search for them in a sound library.

    Returns a list of dicts with the keys in SOUND_KEYS (sound, vibe, search,
    tip). These are AI suggestions based on general short-video audio patterns,
    NOT a real-time trending chart. On repeated failure an empty list is returned
    so the caller can show a friendly message rather than crash.
    """
    amount = _normalize_sound_count(amount)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    language = (language or "").strip()

    language_line = (
        f'Write the "sound", "vibe", "search" and "tip" fields in this language: {language}.'
        if language
        else "Write every text field in the same language as the subject."
    )

    prompt = f"""
# Role: TikTok Affiliate Sound Picker

## Goals:
Suggest {amount} sound / music styles that would fit a short affiliate video about "{video_subject}", and for each give a keyword the creator can search in the app's sound library.

## Important:
1. you cannot see real-time charts, so describe sound STYLES and search keywords, not specific copyrighted track titles you are unsure exist.
2. prefer styles proven to work in short commerce videos: upbeat pop, satisfying/ASMR, calm aesthetic, suspense build-up, trending-style voiceover beats.
3. only suggest royalty-free-friendly or in-app library styles; do not tell the user to rip copyrighted music.

## Constrains:
1. return ONLY a json-array of objects. do not return any text before or after the json.
2. each object must have exactly these keys: "sound", "vibe", "search", "tip".
   - "sound": a short name for the sound / music style.
   - "vibe": the mood or energy it gives the video.
   - "search": a short keyword to find this kind of sound in a sound library.
   - "tip": one short tip on how to use it (e.g. sync the product reveal to the beat).
3. {language_line}
4. make the {amount} suggestions clearly different from each other.

## Output Example:
[{{"sound": "...", "vibe": "...", "search": "...", "tip": "..."}}]
""".strip()

    logger.info(
        f"generating sound ideas: subject={video_subject!r}, amount={amount}, "
        f"language={language!r}"
    )

    ideas = _generate_json_object_list(prompt, _coerce_sound_idea, "sound ideas")
    return ideas[:amount]


DEFAULT_STICKER_COUNT = 5
MAX_STICKER_COUNT = 10
MAX_STICKER_FIELD_LENGTH = 200
STICKER_KEYS = ("text", "timing", "style", "purpose")


def _normalize_sticker_count(amount) -> int:
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return DEFAULT_STICKER_COUNT
    return max(1, min(MAX_STICKER_COUNT, amount))


def _coerce_sticker_idea(item) -> dict:
    """Normalize one LLM-returned text-sticker idea into the fixed {key: str}
    shape, dropping items that are not dicts or carry no on-screen text."""
    if not isinstance(item, dict):
        return {}
    idea = {}
    for key in STICKER_KEYS:
        value = item.get(key, "")
        idea[key] = _clamp_text(value, MAX_STICKER_FIELD_LENGTH)
    if not idea["text"]:
        return {}
    return idea


def generate_text_stickers(
    video_subject: str,
    language: str = "",
    amount: int = DEFAULT_STICKER_COUNT,
) -> List[dict]:
    """Suggest punchy on-screen text overlays / stickers and a clear call-to-action
    for a short affiliate video about ``video_subject``.

    Returns a list of dicts with the keys in STICKER_KEYS (text, timing, style,
    purpose). Each item is an on-screen caption the creator can drop onto the
    video, with when to show it, a quick visual-style hint, and why it helps. On
    repeated failure an empty list is returned so the caller can show a friendly
    message rather than crash.
    """
    amount = _normalize_sticker_count(amount)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    language = (language or "").strip()

    language_line = (
        f'Write the "text", "timing", "style" and "purpose" fields in this language: {language}.'
        if language
        else "Write every text field in the same language as the subject."
    )

    prompt = f"""
# Role: TikTok Affiliate On-screen Text & CTA Writer

## Goals:
Write {amount} short on-screen text stickers (captions the creator overlays on the video) for a short affiliate video about "{video_subject}". Cover the full video arc: at least one scroll-stopping opener, one that builds desire or trust in the middle, and one strong call-to-action near the end.

## Important:
1. on-screen text must be VERY short and punchy — a few words, the way TikTok captions read; do not write full sentences or paragraphs.
2. make at least one item an explicit call-to-action (e.g. tap the link, check the cart, comment a word) without inventing fake discounts, prices, or guarantees.
3. do not promise specific results, medical claims, or anything you cannot back up; keep it honest.

## Constrains:
1. return ONLY a json-array of objects. do not return any text before or after the json.
2. each object must have exactly these keys: "text", "timing", "style", "purpose".
   - "text": the short on-screen caption itself.
   - "timing": when in the video to show it (e.g. first 2 seconds, during the demo, at the end).
   - "style": a quick visual styling hint (e.g. bold yellow top-center, small caption bottom).
   - "purpose": one short note on what it does (hook, build trust, create urgency, call-to-action).
3. {language_line}
4. make the {amount} stickers clearly different from each other.

## Output Example:
[{{"text": "...", "timing": "...", "style": "...", "purpose": "..."}}]
""".strip()

    logger.info(
        f"generating text stickers: subject={video_subject!r}, amount={amount}, "
        f"language={language!r}"
    )

    stickers = _generate_json_object_list(
        prompt, _coerce_sticker_idea, "text stickers"
    )
    return stickers[:amount]


DEFAULT_COVER_COUNT = 4
MAX_COVER_COUNT = 8
MAX_COVER_FIELD_LENGTH = 200
COVER_KEYS = ("text", "subtext", "angle", "tip")


def _normalize_cover_count(amount) -> int:
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return DEFAULT_COVER_COUNT
    return max(1, min(MAX_COVER_COUNT, amount))


def _coerce_cover_idea(item) -> dict:
    """Normalize one LLM-returned cover-text idea into the fixed {key: str}
    shape, dropping items that are not dicts or carry no headline text."""
    if not isinstance(item, dict):
        return {}
    idea = {}
    for key in COVER_KEYS:
        value = item.get(key, "")
        idea[key] = _clamp_text(value, MAX_COVER_FIELD_LENGTH)
    if not idea["text"]:
        return {}
    return idea


def generate_cover_text_ideas(
    video_subject: str,
    language: str = "",
    amount: int = DEFAULT_COVER_COUNT,
) -> List[dict]:
    """Suggest bold cover / thumbnail headlines for a short affiliate video about
    ``video_subject`` — the text shown on the cover frame in the For You / profile
    grid that makes someone tap.

    Returns a list of dicts with the keys in COVER_KEYS (text, subtext, angle,
    tip). ``text`` is the big headline, ``subtext`` an optional smaller second
    line, ``angle`` the hook it leans on, and ``tip`` a quick design/placement
    note. On repeated failure an empty list is returned so the caller can show a
    friendly message rather than crash.
    """
    amount = _normalize_cover_count(amount)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    language = (language or "").strip()

    language_line = (
        f'Write the "text", "subtext", "angle" and "tip" fields in this language: {language}.'
        if language
        else "Write every text field in the same language as the subject."
    )

    prompt = f"""
# Role: TikTok Affiliate Cover / Thumbnail Headline Writer

## Goals:
Write {amount} cover (thumbnail) headline options for a short affiliate video about "{video_subject}". This is the bold text shown on the cover frame in the For You feed and profile grid — its only job is to make someone stop and tap.

## Important:
1. the main "text" headline must be VERY short and punchy — a few big words that read instantly on a small thumbnail; do not write a full sentence.
2. lean on a curiosity, benefit, problem or social-proof angle, but stay honest — no fake prices, fake discounts, or claims you cannot back up.
3. give each option a clearly different angle so the creator can A/B test covers.

## Constrains:
1. return ONLY a json-array of objects. do not return any text before or after the json.
2. each object must have exactly these keys: "text", "subtext", "angle", "tip".
   - "text": the big headline on the cover.
   - "subtext": an optional shorter second line (use "" if none needed).
   - "angle": the hook it leans on (curiosity, benefit, problem, social proof).
   - "tip": one short design/placement tip (e.g. bright frame, keep text in the top third, high contrast).
3. {language_line}
4. make the {amount} options clearly different from each other.

## Output Example:
[{{"text": "...", "subtext": "...", "angle": "...", "tip": "..."}}]
""".strip()

    logger.info(
        f"generating cover text ideas: subject={video_subject!r}, amount={amount}, "
        f"language={language!r}"
    )

    ideas = _generate_json_object_list(prompt, _coerce_cover_idea, "cover text ideas")
    return ideas[:amount]


DEFAULT_SCHEDULE_COUNT = 4
MAX_SCHEDULE_COUNT = 8
MAX_SCHEDULE_FIELD_LENGTH = 200
SCHEDULE_KEYS = ("slot", "day", "time", "why")


def _normalize_schedule_count(amount) -> int:
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return DEFAULT_SCHEDULE_COUNT
    return max(1, min(MAX_SCHEDULE_COUNT, amount))


def _coerce_schedule_slot(item) -> dict:
    """Normalize one LLM-returned posting-time slot into the fixed {key: str}
    shape, dropping items that are not dicts or carry no time window."""
    if not isinstance(item, dict):
        return {}
    slot = {}
    for key in SCHEDULE_KEYS:
        value = item.get(key, "")
        slot[key] = _clamp_text(value, MAX_SCHEDULE_FIELD_LENGTH)
    if not slot["time"]:
        return {}
    return slot


def generate_posting_schedule(
    video_subject: str,
    language: str = "",
    audience_region: str = "",
    amount: int = DEFAULT_SCHEDULE_COUNT,
) -> List[dict]:
    """Suggest best-practice posting time windows for a TikTok affiliate video
    about ``video_subject`` (optionally tuned to ``audience_region``).

    Returns a list of dicts with the keys in SCHEDULE_KEYS (slot, day, time,
    why). These are GENERAL best-practice patterns based on typical short-video
    audience behaviour, NOT the creator's real analytics — the caller should tell
    the user to confirm against their own TikTok analytics. On repeated failure an
    empty list is returned so the caller can show a friendly message rather than
    crash.
    """
    amount = _normalize_schedule_count(amount)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    audience_region = _limit_social_text(
        audience_region, MAX_SOCIAL_SUBJECT_LENGTH, "audience_region"
    )
    language = (language or "").strip()

    language_line = (
        f'Write the "slot", "day", "time" and "why" fields in this language: {language}.'
        if language
        else "Write every text field in the same language as the subject."
    )
    region_line = (
        f"Tune the windows to this audience region / timezone: {audience_region}."
        if audience_region
        else "Assume a general audience and give times in the creator's local time."
    )

    prompt = f"""
# Role: TikTok Affiliate Posting-Time Advisor

## Goals:
Suggest {amount} best-practice posting time windows for a short affiliate video about "{video_subject}". {region_line}

## Important:
1. you CANNOT see the creator's real analytics, so give GENERAL best-practice windows based on typical short-video audience behaviour, and make clear these are starting points to test.
2. base the windows on when shoppers usually browse (commute, lunch break, evening wind-down, weekend), and adapt to the subject/audience where it makes sense.
3. give each window as a day grouping plus a clock time range; keep them clearly different and ordered from strongest to weakest.

## Constrains:
1. return ONLY a json-array of objects. do not return any text before or after the json.
2. each object must have exactly these keys: "slot", "day", "time", "why".
   - "slot": a short label for the window (e.g. Prime evening, Lunch break).
   - "day": which days it applies to (e.g. Weekdays, Sat-Sun, Every day).
   - "time": a clock time range (e.g. 7:00-9:00 PM).
   - "why": one short reason this window tends to work for shoppers.
3. {language_line}
4. make the {amount} windows clearly different from each other.

## Output Example:
[{{"slot": "...", "day": "...", "time": "...", "why": "..."}}]
""".strip()

    logger.info(
        f"generating posting schedule: subject={video_subject!r}, amount={amount}, "
        f"region={audience_region!r}, language={language!r}"
    )

    slots = _generate_json_object_list(
        prompt, _coerce_schedule_slot, "posting schedule"
    )
    return slots[:amount]


DEFAULT_PINNED_COMMENT_COUNT = 4
MAX_PINNED_COMMENT_COUNT = 8
MAX_PINNED_COMMENT_FIELD_LENGTH = 300
PINNED_COMMENT_KEYS = ("comment", "cta", "tip")


def _normalize_pinned_comment_count(amount) -> int:
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return DEFAULT_PINNED_COMMENT_COUNT
    return max(1, min(MAX_PINNED_COMMENT_COUNT, amount))


def _coerce_pinned_comment(item) -> dict:
    """Normalize one LLM-returned pinned-comment idea into the fixed {key: str}
    shape, dropping items that are not dicts or carry no comment text."""
    if not isinstance(item, dict):
        return {}
    pinned = {}
    for key in PINNED_COMMENT_KEYS:
        value = item.get(key, "")
        pinned[key] = _clamp_text(value, MAX_PINNED_COMMENT_FIELD_LENGTH)
    if not pinned["comment"]:
        return {}
    return pinned


def generate_pinned_comments(
    video_subject: str,
    language: str = "",
    amount: int = DEFAULT_PINNED_COMMENT_COUNT,
) -> List[dict]:
    """Write the creator's own first comment to PIN under an affiliate video about
    ``video_subject`` — the pinned comment is prime real estate for the link / CTA
    and for sparking the replies that boost reach.

    Returns a list of dicts with the keys in PINNED_COMMENT_KEYS (comment, cta,
    tip). ``comment`` is the pinned comment text, ``cta`` the short call-to-action
    line it ends on, and ``tip`` a quick note on why/how to use it. This is the
    creator's OWN comment to pin, distinct from generate_comment_replies which
    answers viewers. On repeated failure an empty list is returned so the caller
    can show a friendly message rather than crash.
    """
    amount = _normalize_pinned_comment_count(amount)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    language = (language or "").strip()

    language_line = (
        f'Write the "comment", "cta" and "tip" fields in this language: {language}.'
        if language
        else "Write every text field in the same language as the subject."
    )

    prompt = f"""
# Role: TikTok Affiliate Pinned-Comment Writer

## Goals:
Write {amount} options for the creator's OWN first comment to pin to the top of a short affiliate video about "{video_subject}". A pinned comment is prime real estate: it points viewers to the link, answers the obvious first question, and invites replies that boost reach.

## Important:
1. this is the creator's own comment to pin, NOT a reply to a viewer.
2. each option should hook curiosity or answer the top buying question, then end on a clear call-to-action that points to the link in bio / cart.
3. keep it short and natural — a couple of lines a real creator would type, light emoji is fine.
4. give each option a clearly different angle (link drop, question to spark replies, quick benefit, social proof) so the creator can pick.

## Constrains:
1. return ONLY a json-array of objects. do not return any text before or after the json.
2. each object must have exactly these keys: "comment", "cta", "tip".
   - "comment": the pinned comment text.
   - "cta": the short call-to-action line it ends on (e.g. "Link in bio 👆").
   - "tip": one short note on when/why to use this angle.
3. {language_line}
4. do not invent fake prices, fake discounts, fake reviews, or unverifiable claims; keep it honest and non-spammy.
5. make the {amount} options clearly different from each other.

## Output Example:
[{{"comment": "...", "cta": "...", "tip": "..."}}]
""".strip()

    logger.info(
        f"generating pinned comments: subject={video_subject!r}, amount={amount}, "
        f"language={language!r}"
    )

    pinned = _generate_json_object_list(
        prompt, _coerce_pinned_comment, "pinned comments"
    )
    return pinned[:amount]


DEFAULT_DISCLOSURE_COUNT = 3
MAX_DISCLOSURE_COUNT = 6
MAX_DISCLOSURE_FIELD_LENGTH = 300
DISCLOSURE_KEYS = ("line", "placement", "note")


def _normalize_disclosure_count(amount) -> int:
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return DEFAULT_DISCLOSURE_COUNT
    return max(1, min(MAX_DISCLOSURE_COUNT, amount))


def _coerce_disclosure_line(item) -> dict:
    """Normalize one LLM-returned affiliate-disclosure option into the fixed
    {key: str} shape, dropping items that are not dicts or carry no line text."""
    if not isinstance(item, dict):
        return {}
    disclosure = {}
    for key in DISCLOSURE_KEYS:
        value = item.get(key, "")
        disclosure[key] = _clamp_text(value, MAX_DISCLOSURE_FIELD_LENGTH)
    if not disclosure["line"]:
        return {}
    return disclosure


def generate_disclosure_lines(
    video_subject: str,
    language: str = "",
    amount: int = DEFAULT_DISCLOSURE_COUNT,
) -> List[dict]:
    """Write ready-to-use affiliate / sponsorship DISCLOSURE lines for a short
    video about ``video_subject``. Affiliate and paid content must be disclosed
    (FTC guidance and TikTok/YouTube/Instagram policy), and creators often skip
    or bury it; these give clear, honest options to paste into the caption, an
    on-screen label, the pinned comment, or the spoken intro.

    Returns a list of dicts with the keys in DISCLOSURE_KEYS (line, placement,
    note). ``line`` is the disclosure text, ``placement`` where to use it, and
    ``note`` a short compliance reminder. On repeated failure an empty list is
    returned so the caller can show a friendly message rather than crash.
    """
    amount = _normalize_disclosure_count(amount)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    language = (language or "").strip()

    language_line = (
        f'Write the "line", "placement" and "note" fields in this language: {language}.'
        if language
        else "Write every text field in the same language as the subject."
    )

    prompt = f"""
# Role: Short-Video Affiliate Disclosure & Compliance Writer

## Goals:
Write {amount} ready-to-use affiliate / paid-partnership DISCLOSURE options for a short affiliate video about "{video_subject}". Creators must clearly disclose that they earn from the links, and these should be honest, plain, and easy to paste.

## Important:
1. each "line" must be a clear, conspicuous disclosure a viewer easily understands (e.g. mentions affiliate link / commission / paid partnership / #ad). Do NOT hide or soften the fact that it is sponsored/affiliate.
2. cover a range of placements so the creator can pick: caption text, an on-screen label in the first seconds, the spoken intro, and the pinned comment.
3. keep it short, natural and honest — no fake urgency, fake discounts, or claims about results; disclosure is about transparency, not a sales pitch.
4. include a common hashtag form where natural (e.g. #ad, #affiliate), but the written/spoken sentence must stand on its own without relying only on a hashtag.

## Constrains:
1. return ONLY a json-array of objects. do not return any text before or after the json.
2. each object must have exactly these keys: "line", "placement", "note".
   - "line": the disclosure text to use.
   - "placement": where to put it (e.g. Caption, On-screen first 3s, Spoken intro, Pinned comment).
   - "note": one short compliance reminder (e.g. keep it visible, say it out loud, don't bury it).
3. {language_line}
4. make the {amount} options clearly different in wording and placement.

## Output Example:
[{{"line": "...", "placement": "...", "note": "..."}}]
""".strip()

    logger.info(
        f"generating disclosure lines: subject={video_subject!r}, amount={amount}, "
        f"language={language!r}"
    )

    lines = _generate_json_object_list(
        prompt, _coerce_disclosure_line, "disclosure lines"
    )
    return lines[:amount]


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)
    
