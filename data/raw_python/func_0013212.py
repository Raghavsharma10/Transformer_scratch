def gather_data(registry):
    """Gathers the metrics"""

    # Get the host name of the machine
    host = socket.gethostname()

    # Create our collectors
    trig_metric = Gauge("trigonometry_example",
                        "Various trigonometry examples.",
                        {'host': host})

    # register the metric collectors
    registry.register(trig_metric)

    # Start gathering metrics every second
    counter = 0
    while True:
        time.sleep(1)

        sine = math.sin(math.radians(counter % 360))
        cosine = math.cos(math.radians(counter % 360))
        trig_metric.set({'type': "sine"}, sine)
        trig_metric.set({'type': "cosine"}, cosine)

        counter += 1