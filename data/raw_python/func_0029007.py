def _remove_event_source(awsclient, evt_source, lambda_arn):
    """
    Given an event_source dictionary, create the object and remove the event source.
    """
    event_source_obj = _get_event_source_obj(awsclient, evt_source)
    if event_source_obj.exists(lambda_arn):
        event_source_obj.remove(lambda_arn)