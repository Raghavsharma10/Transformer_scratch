def _runable_for_event(f, tag, stage):
    """Loot at the event property for a function to see if it should be run at this stage. """

    if not hasattr(f, '__ambry_event__'):
        return False

    f_tag, f_stage = f.__ambry_event__

    if stage is None:
        stage = 0

    if tag != f_tag or stage != f_stage:
        return False

    return True