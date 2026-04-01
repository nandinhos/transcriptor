import streamlit as st
import whisper
import tempfile
import os
import time
from pathlib import Path
from lib.model_manager import get_model_info, is_model_downloaded, load_model_with_check
from lib.queue_manager import QueueManager, JobStatus
from lib.embeddings import EmbeddingsManager

st.set_page_config(page_title="Transcritor Pro", page_icon="🎙️", layout="wide")

st.markdown(
    """
    <style>
        .stFileUploader [data-testid="stFileUploader"] {
            max-upload-size-mb: 2000;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Configurar limite de upload para 2GB
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

queue = QueueManager("transcripts")
embeddings_mgr = EmbeddingsManager()


def load_model_cached(model_name, device="cpu"):
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
        st.session_state.model_obj = None

    if st.session_state.current_model != model_name:
        with st.spinner(f"Carregando modelo {model_name}..."):
            st.session_state.model_obj = load_model_with_check(model_name, device)
            st.session_state.current_model = model_name

    return st.session_state.model_obj


def process_audio(job_id, file_path, model_name):
    try:
        queue.update_status(job_id, JobStatus.PROCESSING.value)
        model = load_model_cached(model_name)

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

    tab1, tab2, tab3 = st.tabs(
        ["📤 Upload & Fila", "📝 Validação", "💬 Chat de Estudos"]
    )

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Upload de Arquivos")
            uploaded_files = st.file_uploader(
                "Selecione arquivos de áudio/vídeo (máx 2GB por arquivo)",
                type=["mp4", "mp3", "wav", "m4a", "ogg", "flac", "avi", "mkv"],
                accept_multiple_files=True,
                max_upload_size=2000,
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
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uploaded_file.name).suffix
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        text, text_path = process_audio(
                            job_id, tmp_path, selected_model
                        )
                        st.success(f"Job {job_id}: Transcrição concluída!")
                    except Exception as e:
                        st.error(f"Erro: {e}")
                    finally:
                        os.unlink(tmp_path)

        st.divider()
        st.subheader("Fila de Execução")
        queue_data = queue.get_queue()

        if queue_data:
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
        st.markdown("Faça perguntas sobre o conteúdo transcrito e vetorizado")

        embedded_count = len(embeddings_mgr.get_all_embedded())
        st.write(f"📚 {embedded_count} chunks vetorizados")

        if embedded_count > 0:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            query = st.chat_input("Digite sua pergunta sobre o conteúdo...")

            if query:
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)

                with st.chat_message("assistant"):
                    with st.spinner("Buscando informações..."):
                        results = embeddings_mgr.search(query, top_k=5)

                    if results:
                        response = "Encontrei as seguintes informações:\n\n"
                        for r in results:
                            response += f"📄 **{r['filename']}**:\n{r['text']}\n\n"

                        st.markdown(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                    else:
                        st.warning("Não encontrei informações relevantes")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": "Não encontrei informações relevantes nos documentos.",
                            }
                        )

            if st.button("🗑️ Limpar Chat"):
                st.session_state.messages = []
                st.rerun()
        else:
            st.warning(
                "Nenhum conteúdo vetorizado ainda. Valide uma transcrição e gere embeddings primeiro."
            )


if __name__ == "__main__":
    main()
