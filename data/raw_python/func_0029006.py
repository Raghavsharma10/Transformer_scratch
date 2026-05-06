def _add_event_source(awsclient, evt_source, lambda_arn):
    """
    Given an event_source dictionary, create the object and add the event source.
    """
    event_source_obj = _get_event_source_obj(awsclient, evt_source)

    # (where zappa goes like remove, add)
    # we go with update and add like this:
    if event_source_obj.exists(lambda_arn):
        event_source_obj.update(lambda_arn)
    else:
        event_source_obj.add(lambda_arn)