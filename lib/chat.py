import json
import requests
from typing import List, Generator

OLLAMA_BASE_URL = "http://localhost:11434"

SYSTEM_PROMPT = """Você é um assistente de estudos inteligente. Seu papel é ajudar o usuário a entender e explorar o conteúdo de aulas, palestras e vídeos que foram transcritos.

Quando houver contexto relevante disponível, baseie sua resposta nele. Quando a pergunta for de cunho geral ou conceitual, responda com seu próprio conhecimento de forma didática.

Se o usuário fizer uma pergunta sobre algo que claramente não está no conteúdo transcrito e também não for um conceito geral que você conheça, informe isso com clareza.

CONTEXTO DO CONTEÚDO TRANSCRITO:
{context}
"""


def get_ollama_models() -> List[str]:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


def is_ollama_running() -> bool:
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def stream_chat(messages: List[dict], context_chunks: List[dict], model: str) -> Generator[str, None, None]:
    if context_chunks:
        context = "\n\n".join(
            f"[{c['filename']}]: {c['text']}" for c in context_chunks
        )
    else:
        context = "Nenhum trecho relevante encontrado no conteúdo transcrito para esta pergunta."

    system_msg = {"role": "system", "content": SYSTEM_PROMPT.format(context=context)}
    payload = {
        "model": model,
        "messages": [system_msg] + messages,
        "stream": True,
    }

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            delta = data.get("message", {}).get("content", "")
            if delta:
                yield delta
            if data.get("done"):
                break
