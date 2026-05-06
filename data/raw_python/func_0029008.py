def _get_event_source_status(awsclient, evt_source, lambda_arn):
    """
    Given an event_source dictionary, create the object and get the event source status.
    """
    event_source_obj = _get_event_source_obj(awsclient, evt_source)
    return event_source_obj.status(lambda_arn)