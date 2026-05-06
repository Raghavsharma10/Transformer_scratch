def gather_data(registry):
    """Gathers the metrics"""

    # Get the host name of the machine
    host = socket.gethostname()

    # Create our collectors
    ram_metric = Gauge("memory_usage_bytes", "Memory usage in bytes.",
                       {'host': host})
    cpu_metric = Gauge("cpu_usage_percent", "CPU usage percent.",
                       {'host': host})

    # register the metric collectors
    registry.register(ram_metric)
    registry.register(cpu_metric)

    # Start gathering metrics every second
    while True:
        time.sleep(1)

        # Add ram metrics
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()

        ram_metric.set({'type': "virtual", }, ram.used)
        ram_metric.set({'type': "virtual", 'status': "cached"}, ram.cached)
        ram_metric.set({'type': "swap"}, swap.used)

        # Add cpu metrics
        for c, p in enumerate(psutil.cpu_percent(interval=1, percpu=True)):
            cpu_metric.set({'core': c}, p)