# app/worker.py
import redis
from rq import Queue, Worker
from app.services.docling_service import DoclingService
import os

def process_pdf(file_path: str):
    service = DoclingService(file_path)
    return service.docling_funtion()

if __name__ == "__main__":
    redis_conn = redis.Redis(host="localhost", port=6379)
    worker = Worker([Queue("docling", connection=redis_conn)], connection=redis_conn)
    worker.work()