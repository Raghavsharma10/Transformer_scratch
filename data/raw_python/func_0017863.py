def collect_logs(name):
    """
    Returns a string representation of the logs from a container.
    This is similar to container_logs but uses the `follow` option
    and flattens the logs into a string instead of a generator.

    :param name: The container name to grab logs for
    :return: A string representation of the logs
    """
    logs = container_logs(name, "all", True, None)
    string = ""
    for s in logs:
        string += s
    return string