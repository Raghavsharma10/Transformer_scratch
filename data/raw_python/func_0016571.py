def ping(config_file, profile, solver_def, json_output, request_timeout, polling_timeout):
    """Ping the QPU by submitting a single-qubit problem."""

    now = utcnow()
    info = dict(datetime=now.isoformat(), timestamp=datetime_to_timestamp(now), code=0)

    def output(fmt, **kwargs):
        info.update(kwargs)
        if not json_output:
            click.echo(fmt.format(**kwargs))

    def flush():
        if json_output:
            click.echo(json.dumps(info))

    try:
        _ping(config_file, profile, solver_def, request_timeout, polling_timeout, output)
    except CLIError as error:
        output("Error: {error} (code: {code})", error=str(error), code=error.code)
        sys.exit(error.code)
    except Exception as error:
        output("Unhandled error: {error}", error=str(error))
        sys.exit(127)
    finally:
        flush()