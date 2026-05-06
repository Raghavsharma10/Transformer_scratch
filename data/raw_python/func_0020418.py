def graphite(context, server, port, interval, prefix):
    """Display energy stats of all actors"""
    fritz = context.obj
    fritz.login()
    sid_ttl = time.time() + 600

    # Find actors and create carbon keys
    click.echo(" * Requesting actors list")
    simple_chars = re.compile('[^A-Za-z0-9]+')
    actors = fritz.get_actors()
    keys = {}
    for actor in actors:
        keys[actor.name] = "{}.{}".format(
            prefix,
            simple_chars.sub('_', actor.name)
        )

    # Connect to carbon
    click.echo(" * Trying to connect to carbon")
    timeout = 2
    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((server, port))
    except socket.timeout:
        raise Exception("Took over {} second(s) to connect to {}".format(
            timeout, server
        ))
    except Exception as error:
        raise Exception("unknown exception while connecting to {} - {}".format(
            server, error
        ))

    def send(key, value):
        """Send a key-value-pair to carbon"""
        now = int(time.time())
        payload = "{} {} {}\n".format(key, value, now)
        sock.sendall(payload)

    while True:
        if time.time() > sid_ttl:
            click.echo(" * Requesting new SID")
            fritz.login()
            sid_ttl = time.time() + 600

        click.echo(" * Requesting statistics")
        for actor in actors:
            power = actor.get_power()
            total = actor.get_energy()
            click.echo("   -> {}: {:.2f} Watt current, {:.3f} wH total".format(
                actor.name, power / 1000, total / 100
            ))

            send(keys[actor.name] + '.current', power)
            send(keys[actor.name] + '.total', total)

        time.sleep(interval)