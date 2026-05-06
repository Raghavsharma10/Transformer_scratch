def gather_data(registry):
    """Gathers the metrics"""

    # Get the host name of the machine
    host = socket.gethostname()

    # Create our collectors
    io_metric = Summary("write_file_io_example",
                        "Writing io file in disk example.",
                        {'host': host})

    # register the metric collectors
    registry.register(io_metric)
    chunk = b'\xff'*4000  # 4000 bytes
    filename_path = "/tmp/prometheus_test"
    blocksizes = (100, 10000, 1000000, 100000000)

    # Start gathering metrics every 0.7 seconds
    while True:
        time.sleep(0.7)

        for i in blocksizes:
            time_start = time.time()
            # Action
            with open(filename_path, "wb") as f:
                for _ in range(i // 10000):
                    f.write(chunk)

            io_metric.add({"file": filename_path, "block": i},
                          time.time() - time_start)