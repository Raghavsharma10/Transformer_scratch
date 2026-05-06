def _capture_snapshot(a_snapshot: Snapshot, resolved_kwargs: Mapping[str, Any]) -> Any:
    """
    Capture the snapshot from the keyword arguments resolved before the function call (including the default values).

    :param a_snapshot: snapshot to be captured
    :param resolved_kwargs: resolved keyword arguments (including the default values)
    :return: captured value
    """
    if a_snapshot.arg is not None:
        if a_snapshot.arg not in resolved_kwargs:
            raise TypeError(("The argument of the snapshot has not been set: {}. "
                             "Does the original function define it? Did you supply it in the call?").format(
                                 a_snapshot.arg))

        value = a_snapshot.capture(**{a_snapshot.arg: resolved_kwargs[a_snapshot.arg]})
    else:
        value = a_snapshot.capture()

    return value