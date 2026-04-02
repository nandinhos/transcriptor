import warnings

warnings.filterwarnings("ignore", message="Accessing `__path__`")

import streamlit as st
import whisper
import tempfile
import os
import time
from pathlib import Path
from lib.model_manager import (
    get_model_info,
    is_model_downloaded,
    load_model_with_check,
    download_model,
    delete_model,
    get_model_size_on_disk,
    MODEL_SIZES_MB,
)
from lib.queue_manager import QueueManager, JobStatus
from lib.embeddings import EmbeddingsManager
from lib.chat import get_ollama_models, is_ollama_running, stream_chat

st.set_page_config(page_title="Transcritor Pro", page_icon="🎙️", layout="wide")


queue = QueueManager("transcripts")
embeddings_mgr = EmbeddingsManager()


def process_audio(job_id, file_path, model_name):
    try:
        queue.update_status(job_id, JobStatus.PROCESSING.value)
        with st.spinner(f"Transcrevendo com {model_name}..."):
            model = load_model_with_check(model_name)

            result = model.transcribe(file_path, language="pt")
            text = result["text"]

            text_path = queue.save_transcription(job_id, text)
            return text, text_path
    except Exception as e:
        queue.update_status(job_id, JobStatus.ERROR.value, error=str(e))
        raise e


def main():
    st.title("🎙️ Transcritor Pro")
    st.markdown("Transcrição de áudio com fila de execução e chat de estudos")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📤 Upload & Fila", "📝 Validação", "💬 Chat de Estudos", "⚙️ Modelos"]
    )

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Upload de Arquivos")
            uploaded_files = st.file_uploader(
                "Selecione arquivos de áudio/vídeo (máx 2GB por arquivo)",
                type=["mp4", "mp3", "wav", "m4a", "ogg", "flac", "avi", "mkv"],
                accept_multiple_files=True,
            )

        with col2:
            st.subheader("Informações do Modelo")
            model_info = get_model_info()
            st.write(f"**Modelos disponíveis:** {', '.join(model_info['available'])}")
            st.write(
                f"**Modelos baixados:** {', '.join(model_info['downloaded']) if model_info['downloaded'] else 'Nenhum'}"
            )
            st.write(f"**Cache:** `{model_info['cache_dir']}`")

        col_m, col_f = st.columns(2)
        with col_m:
            selected_model = st.selectbox(
                "Modelo Whisper", model_info["available"], index=2
            )

        if uploaded_files:
            st.subheader("Arquivos na Fila")
            for uploaded_file in uploaded_files:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.write(f"📄 **{uploaded_file.name}** ({file_size_mb:.1f} MB)")

                if st.button(
                    f"Adicionar {uploaded_file.name[:15]}...",
                    key=f"add_{uploaded_file.name}",
                ):
                    job_id = queue.add_job(
                        uploaded_file.name, uploaded_file.size, selected_model
                    )
                    file_ext = Path(uploaded_file.name).suffix
                    stored_path = str(queue.storage_path / f"{job_id}_source{file_ext}")
                    with open(stored_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    queue.update_status(job_id, "pendente", file_path=stored_path)

                    try:
                        text, text_path = process_audio(
                            job_id, stored_path, selected_model
                        )
                        st.success(f"Job {job_id}: Transcrição concluída!")
                    except Exception as e:
                        st.error(f"Erro: {e}")

        st.divider()
        st.subheader("Fila de Execução")
        queue_data = queue.get_queue()

        if queue_data:
            unexecuted = [j for j in queue_data if j["status"] in ("pendente", "erro")]
            if unexecuted:
                if st.button(f"🗑️ Limpar não executados ({len(unexecuted)})"):
                    queue.clear_unexecuted()
                    st.rerun()

            for job in queue_data:
                status_emoji = {
                    "pendente": "⏳",
                    "processando": "⚙️",
                    "validando": "✏️",
                    "concluido": "✅",
                    "vetorizado": "🔎",
                    "erro": "❌",
                }.get(job["status"], "❓")

                with st.expander(f"{status_emoji} {job['filename']} - {job['status']}"):
                    st.write(f"**ID:** {job['id']}")
                    st.write(f"**Modelo:** {job['model']}")
                    st.write(f"**Tamanho:** {job['file_size'] / (1024 * 1024):.1f} MB")
                    st.write(f"**Criado:** {job['created_at']}")

                    confirm_key = f"confirm_delete_{job['id']}"
                    if st.session_state.get(confirm_key):
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button(
                                "Confirmar exclusão", key=f"confirm_yes_{job['id']}"
                            ):
                                queue.delete_job(job["id"])
                                st.session_state.pop(confirm_key, None)
                                st.rerun()
                        with c2:
                            if st.button("Cancelar", key=f"confirm_no_{job['id']}"):
                                st.session_state.pop(confirm_key, None)
                                st.rerun()
                    else:
                        if st.button("🗑️ Excluir", key=f"delete_{job['id']}"):
                            st.session_state[confirm_key] = True
                            st.rerun()

                    if job["status"] == "erro":
                        st.error(f"Erro: {job.get('error', 'desconhecido')}")
                        retry_model = st.selectbox(
                            "Modelo para retry",
                            model_info["available"],
                            index=model_info["available"].index(job["model"])
                            if job["model"] in model_info["available"]
                            else 2,
                            key=f"retry_model_{job['id']}",
                        )
                        if st.button("🔄 Tentar novamente", key=f"retry_{job['id']}"):
                            stored = job.get("file_path")
                            if stored and Path(stored).exists():
                                queue.reset_for_retry(job["id"], retry_model)
                                try:
                                    process_audio(job["id"], stored, retry_model)
                                    st.success("Transcrição concluída!")
                                except Exception as e:
                                    st.error(f"Erro: {e}")
                                st.rerun()
                            else:
                                st.warning(
                                    "Arquivo original não encontrado. Faça o upload novamente."
                                )

                    if job.get("transcription"):
                        st.text_area(
                            "Transcrição",
                            job["transcription"],
                            height=150,
                            key=f"text_{job['id']}",
                        )
                        if job["status"] == "validando":
                            c1, c2 = st.columns(2)
                            with c1:
                                if st.button("✅ Validar", key=f"val_{job['id']}"):
                                    queue.mark_validated(job["id"])
                                    st.rerun()
                            with c2:
                                if st.button(
                                    "🔎 Gerar Embeddings", key=f"emb_{job['id']}"
                                ):
                                    try:
                                        embeddings_mgr.add_transcription(
                                            job["id"],
                                            job["transcription"],
                                            job["filename"],
                                        )
                                        queue.mark_embedded(job["id"])
                                        st.success("Embeddings gerados!")
                                    except Exception as e:
                                        st.error(f"Erro: {e}")
                                    st.rerun()
        else:
            st.info("Nenhum arquivo na fila")

    with tab2:
        st.subheader("Transcrições Validadas")
        validated_jobs = [
            j
            for j in queue.get_queue()
            if j["status"] in ["validando", "concluido", "vetorizado"]
        ]

        if validated_jobs:
            for job in validated_jobs:
                with st.expander(f"📝 {job['filename']}"):
                    if job.get("text_path") and Path(job["text_path"]).exists():
                        with open(job["text_path"], "r") as f:
                            content = f.read()
                        st.text_area(
                            "Texto", content, height=200, key=f"edit_{job['id']}"
                        )
                    else:
                        st.text_area(
                            "Texto",
                            job.get("transcription", ""),
                            height=200,
                            key=f"edit_{job['id']}",
                        )

                    c1, c2 = st.columns(2)
                    with c1:
                        status_icon = (
                            "✅"
                            if job["status"] == "concluido"
                            or job["status"] == "vetorizado"
                            else "✏️"
                        )
                        st.write(f"**Status:** {status_icon} {job['status']}")
                    with c2:
                        if job.get("embedded"):
                            st.success("Vetorizado 🔎")
                        elif job["status"] == "concluido":
                            if st.button("Gerar Embeddings", key=f"emb2_{job['id']}"):
                                try:
                                    text = job.get("transcription", "")
                                    if not text and job.get("text_path"):
                                        with open(job["text_path"], "r") as f:
                                            text = f.read()
                                    embeddings_mgr.add_transcription(
                                        job["id"], text, job["filename"]
                                    )
                                    queue.mark_embedded(job["id"])
                                    st.success("Embeddings gerados!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Erro: {e}")
        else:
            st.info("Nenhuma transcrição validada")

    with tab3:
        st.subheader("💬 Chat de Estudos")

        if not is_ollama_running():
            st.error("Ollama não está rodando. Inicie com `ollama serve` no terminal.")
            st.stop()

        ollama_models = get_ollama_models()
        if not ollama_models:
            st.error(
                "Nenhum modelo disponível no Ollama. Execute `ollama pull <modelo>`."
            )
            st.stop()

        col_model, col_clear = st.columns([3, 1])
        with col_model:
            default_idx = next(
                (i for i, m in enumerate(ollama_models) if "minimax" in m.lower()), 0
            )
            chat_model = st.selectbox(
                "Modelo", ollama_models, index=default_idx, key="chat_model"
            )
        with col_clear:
            st.write("")
            if st.button("🗑️ Limpar chat"):
                st.session_state.messages = []
                st.rerun()

        embedded_count = len(embeddings_mgr.get_all_embedded())
        if embedded_count > 0:
            st.caption(
                f"📚 {embedded_count} chunks vetorizados disponíveis como contexto"
            )
        else:
            st.caption(
                "💡 Sem conteúdo vetorizado — o assistente responderá com conhecimento geral"
            )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        messages_container = st.container()
        query = st.chat_input(
            "Pergunte sobre o conteúdo ou qualquer dúvida de estudo..."
        )

        with messages_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if query:
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)

                context_chunks = (
                    embeddings_mgr.search(query, top_k=5) if embedded_count > 0 else []
                )

                with st.chat_message("assistant"):
                    try:
                        response = st.write_stream(
                            stream_chat(
                                st.session_state.messages[:-1],
                                context_chunks,
                                chat_model,
                            )
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                    except Exception as e:
                        st.error(f"Erro ao chamar o modelo: {e}")

    with tab4:
        st.subheader("Gerenciamento de Modelos Whisper")

        active_model = st.session_state.get("current_model")

        for model_name in model_info["available"]:
            downloaded = is_model_downloaded(model_name)
            size_mb = MODEL_SIZES_MB.get(model_name, 0)
            is_active = active_model == model_name

            col_name, col_size, col_status, col_action = st.columns([2, 1, 1, 2])

            with col_name:
                st.write(f"**{model_name}**")
            with col_size:
                if downloaded:
                    real_mb = get_model_size_on_disk(model_name) / (1024 * 1024)
                    st.write(f"{real_mb:.0f} MB")
                else:
                    st.write(f"~{size_mb} MB")
            with col_status:
                if is_active:
                    st.success("em uso")
                elif downloaded:
                    st.info("baixado")
                else:
                    st.caption("não baixado")
            with col_action:
                if not downloaded:
                    if st.button("⬇️ Baixar", key=f"dl_{model_name}"):
                        with st.spinner(f"Baixando {model_name}..."):
                            try:
                                download_model(model_name)
                                st.success(f"{model_name} baixado!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erro: {e}")
                elif is_active:
                    st.button("🔒 Em uso", key=f"inuse_{model_name}", disabled=True)
                else:
                    confirm_key = f"confirm_del_model_{model_name}"
                    if st.session_state.get(confirm_key):
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("Confirmar", key=f"del_yes_{model_name}"):
                                delete_model(model_name)
                                st.session_state.pop(confirm_key, None)
                                st.rerun()
                        with c2:
                            if st.button("Cancelar", key=f"del_no_{model_name}"):
                                st.session_state.pop(confirm_key, None)
                                st.rerun()
                    else:
                        if st.button("🗑️ Excluir", key=f"del_{model_name}"):
                            st.session_state[confirm_key] = True
                            st.rerun()

            st.divider()


if __name__ == "__main__":
    main()
