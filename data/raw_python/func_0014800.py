def _get_base_name(hostname, step_name, attempt_id, max_length):
    """Create a base name for the worker instance that will run the specified
    task run attempt, from this server. Since hostname and step name will be
    duplicated across workers (reruns, etc.), ensure that at least
    MIN_TASK_ID_CHARS are preserved in the instance name. Also, prevent names
    from ending with dashes.
    """
    max_length = int(max_length)
    if len(hostname)+len(step_name)+MIN_TASK_ID_CHARS+2 > max_length:
        # round with ceil/floor such that extra char goes to hostname if odd
        hostname_chars = int(math.ceil(
            (max_length-MIN_TASK_ID_CHARS-2)/float(2)))
        step_name_chars = int(math.floor(
            (max_length-MIN_TASK_ID_CHARS-2)/float(2)))
        hostname = hostname[:hostname_chars]
        step_name = step_name[:step_name_chars]
    name_base = '-'.join([hostname, step_name, attempt_id])
    return _sanitize_instance_name(name_base, max_length)