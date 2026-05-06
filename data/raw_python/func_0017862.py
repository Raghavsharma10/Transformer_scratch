def container_logs(name, tail, follow, timestamps):
    """
    Wrapper for docker logs, attach commands.
    """
    if follow:
        return _get_docker().attach(
            name,
            stdout=True,
            stderr=True,
            stream=True
            )

    return _docker.logs(
        name,
        stdout=True,
        stderr=True,
        tail=tail,
        timestamps=timestamps,
    )