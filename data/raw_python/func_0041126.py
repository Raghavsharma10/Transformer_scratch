def _min_timezone_offset():
    "time zone offset (minutes)"
    now = time.time()
    return (datetime.datetime.fromtimestamp(now) - datetime.datetime.utcfromtimestamp(now)).seconds // 60