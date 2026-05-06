def is_sleep(key):
    """
    Determine return data by use cache if this key is in the sleep time window(happened error)
    """
    lock.acquire()
    try:
        if key not in sleep_record:
            return False
        return time.time() < sleep_record[key]
    finally:
        lock.release()