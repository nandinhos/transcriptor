import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Optional


class JobStatus(Enum):
    PENDING = "pendente"
    PROCESSING = "processando"
    COMPLETED = "concluido"
    ERROR = "erro"
    VALIDATING = "validando"
    EMBEDDED = "vetorizado"


class QueueManager:
    def __init__(self, storage_path: str = "transcripts"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.queue_file = self.storage_path / "queue.json"
        self._init_queue()

    def _init_queue(self):
        if not self.queue_file.exists():
            self._save_queue([])

    def _load_queue(self) -> list:
        try:
            with open(self.queue_file, "r") as f:
                return json.load(f)
        except:
            return []

    def _save_queue(self, queue: list):
        with open(self.queue_file, "w") as f:
            json.dump(queue, f, indent=2, default=str)

    def add_job(self, filename: str, file_size: int, model: str = "small", file_path: str = None) -> str:
        job_id = str(uuid.uuid4())[:8]
        job = {
            "id": job_id,
            "filename": filename,
            "file_size": file_size,
            "model": model,
            "status": JobStatus.PENDING.value,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "transcription": None,
            "text_path": None,
            "file_path": file_path,
            "embedded": False,
        }
        queue = self._load_queue()
        queue.append(job)
        self._save_queue(queue)
        return job_id

    def update_status(self, job_id: str, status: str, **kwargs):
        queue = self._load_queue()
        for job in queue:
            if job["id"] == job_id:
                job["status"] = status
                if status == JobStatus.PROCESSING.value:
                    job["started_at"] = datetime.now().isoformat()
                elif status == JobStatus.COMPLETED.value:
                    job["completed_at"] = datetime.now().isoformat()
                for key, value in kwargs.items():
                    job[key] = value
                break
        self._save_queue(queue)

    def get_queue(self) -> list:
        return self._load_queue()

    def get_job(self, job_id: str) -> Optional[dict]:
        queue = self._load_queue()
        for job in queue:
            if job["id"] == job_id:
                return job
        return None

    def get_pending_jobs(self) -> list:
        queue = self._load_queue()
        return [j for j in queue if j["status"] == JobStatus.PENDING.value]

    def get_next_pending(self) -> Optional[dict]:
        pending = self.get_pending_jobs()
        return pending[0] if pending else None

    def get_job_by_filename(self, filename: str) -> Optional[dict]:
        queue = self._load_queue()
        for job in queue:
            if job["filename"] == filename:
                return job
        return None

    def clear_completed(self):
        queue = self._load_queue()
        queue = [
            j
            for j in queue
            if j["status"] != JobStatus.COMPLETED.value
            and j["status"] != JobStatus.ERROR.value
        ]
        self._save_queue(queue)

    def save_transcription(self, job_id: str, text: str) -> Path:
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} não encontrado")

        text_path = self.storage_path / f"{job_id}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

        self.update_status(
            job_id,
            JobStatus.VALIDATING.value,
            transcription=text,
            text_path=str(text_path),
        )
        return text_path

    def mark_validated(self, job_id: str):
        self.update_status(job_id, JobStatus.COMPLETED.value)

    def mark_embedded(self, job_id: str):
        self.update_status(job_id, JobStatus.EMBEDDED.value, embedded=True)

    def delete_job(self, job_id: str):
        queue = self._load_queue()
        job = next((j for j in queue if j["id"] == job_id), None)
        if job:
            file_path = job.get("file_path")
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
        self._save_queue([j for j in queue if j["id"] != job_id])

    def clear_unexecuted(self):
        queue = self._load_queue()
        removable = [
            j for j in queue
            if j["status"] in (JobStatus.PENDING.value, JobStatus.ERROR.value)
        ]
        for job in removable:
            file_path = job.get("file_path")
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
        self._save_queue([j for j in queue if j not in removable])

    def reset_for_retry(self, job_id: str, model: str):
        queue = self._load_queue()
        for job in queue:
            if job["id"] == job_id:
                job["status"] = JobStatus.PENDING.value
                job["model"] = model
                job["error"] = None
                job["transcription"] = None
                job["text_path"] = None
                job["started_at"] = None
                job["completed_at"] = None
                break
        self._save_queue(queue)
