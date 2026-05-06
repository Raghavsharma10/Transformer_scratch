def stream_replicate():
    """Monitor changes in approximately real-time and replicate them"""
    stream = primary.stream(SomeDataBlob, "trim_horizon")
    next_heartbeat = pendulum.now()
    while True:
        now = pendulum.now()
        if now >= next_heartbeat:
            stream.heartbeat()
            next_heartbeat = now.add(minutes=10)

        record = next(stream)
        if record is None:
            continue
        if record["new"] is not None:
            replica.save(record["new"])
        else:
            replica.delete(record["old"])