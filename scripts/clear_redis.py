from dependencies.cache import cache

def main():
    """
    Flush *only* the Redis DB defined by REDIS_CACHE_DB in your .env.
    """
    count = cache.flushdb()
    print(f"✔️  Flushed Redis cache DB ({count} keys removed)")

if __name__ == "__main__":
    main()