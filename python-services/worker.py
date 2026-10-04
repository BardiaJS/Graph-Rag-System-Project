#!/usr/bin/env python3
"""
Worker for processing PDF documents in the background
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

import redis
from dotenv import load_dotenv
import logging

# تنظیم لاگینگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# بارگذاری متغیرهای محیطی
load_dotenv()

# تنظیمات Redis
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'decode_responses': True,
    'socket_connect_timeout': 10,
    'socket_timeout': 10,
    'health_check_interval': 30,
}

logger.info(f"Connecting to Redis at {REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}")

# اتصال به Redis با تلاش مجدد
def get_redis_connection():
    """ایجاد اتصال به Redis با تلاش مجدد"""
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            client = redis.Redis(**REDIS_CONFIG)
            client.ping()
            logger.info(f"✅ Connected to Redis successfully (attempt {attempt + 1})")
            return client
        except redis.ConnectionError as e:
            logger.warning(f"⚠️ Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

# ایجاد اتصال Redis در سطح جهانی
try:
    redis_client = get_redis_connection()
except Exception as e:
    logger.error(f"❌ Failed to connect to Redis: {e}")
    sys.exit(1)

# import services بعد از تنظیم مسیر
from app.services.document_processor import DocumentProcessor

async def process_queue():
    """پردازش صف اسناد"""
    logger.info("🚀 Worker started, waiting for documents...")
    logger.info(f"📡 Connected to Redis at {REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}")
    
    processor = DocumentProcessor()
    
    # استفاده از redis_client تعریف شده در سطح بالا
    global redis_client
    
    while True:
        try:
            # گرفتن از صف با timeout کمتر
            result = redis_client.blpop('document_queue', timeout=1)
            
            if result:
                _, data = result
                try:
                    event = json.loads(data)
                    logger.info(f"\n📄 Processing document: {event.get('file_path', 'unknown')}")
                    logger.info(f"   User ID: {event.get('user_id')}")
                    logger.info(f"   Session ID: {event.get('session_id')}")
                    logger.info(f"   Document IDs: {event.get('document_ids', [])}")
                    
                    # پردازش سند
                    result = await processor.process(
                        file_path=event['file_path'],
                        user_id=event['user_id'],
                        session_id=event['session_id'],
                        document_ids=event['document_ids']
                    )
                    
                    logger.info(f"✅ Document processed successfully!")
                    logger.info(f"   Result: {result}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON data: {e}")
                except Exception as e:
                    logger.error(f"❌ Error processing document: {e}")
                    import traceback
                    traceback.print_exc()
            
            # اطمینان از اینکه اتصال Redis هنوز برقرار است
            try:
                redis_client.ping()
            except redis.ConnectionError:
                logger.warning("⚠️ Redis connection lost. Reconnecting...")
                redis_client = get_redis_connection()
                
        except redis.ConnectionError as e:
            logger.error(f"❌ Redis connection error: {e}")
            logger.info("🔄 Attempting to reconnect...")
            try:
                redis_client = get_redis_connection()
            except Exception as reconnect_error:
                logger.error(f"❌ Reconnection failed: {reconnect_error}")
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(process_queue())
    except KeyboardInterrupt:
        logger.info("\n🛑 Worker stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)