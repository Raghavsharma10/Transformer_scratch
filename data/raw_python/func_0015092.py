def emitMany(*args, **kwargs):
    """A more efficient way to emit a number of tuples at once."""
    global MODE
    if MODE == Bolt:
        emitManyBolt(*args, **kwargs)
    elif MODE == Spout:
        emitManySpout(*args, **kwargs)