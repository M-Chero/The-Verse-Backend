import os
import redis
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

# REDIS_HOST = os.getenv("REDIS_HOST")
# REDIS_PORT = int(os.getenv("REDIS_PORT"))
# REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
# REDIS_DB = int(os.getenv("REDIS_CACHE_DB", 1))

CACHE_TTL = int(os.getenv("SESSION_LIFETIME", 120)) * 60 

# cache = redis.Redis(
#     host=REDIS_HOST,
#     port=REDIS_PORT,
#     password=REDIS_PASSWORD,
#     db=REDIS_DB,
#     decode_responses=True,
# )

url = urlparse(os.environ.get("REDIS_URL"))
cache=redis.Redis(host=url.hostname, port=url.port, password=url.password, ssl=(url.scheme == "rediss"), ssl_cert_reqs=None)