def sendFailureMsgToParent(msg):
    """This function is kind of a hack, but useful when a Python task
    encounters a fatal exception. "msg" should be a simple string like
    "E_SPOUTFAILED". This function sends "msg" as-is to the Storm worker,
    which tries to parse it as JSON. The hacky aspect is that we
    *deliberately* make it fail by sending it non-JSON data. This causes
    the Storm worker to throw an error and restart the Python task. This
    is cleaner than simply letting the task die without notifying Storm,
    because this way Storm restarts the task more quickly."""
    assert isinstance(msg, six.string_types)
    print(msg, file=old_stdout)
    print('end', file=old_stdout)
    storm_log.error('Sent failure message ("%s") to Storm', msg)