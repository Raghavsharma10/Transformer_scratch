def create_event_model(event):
    """ Factory function that turns a celery event into an event object.

    Args:
        event (dict): A dictionary that represents a celery event.

    Returns:
        object: An event object representing the received event.

    Raises:
        JobEventTypeUnsupported: If an unsupported celery job event was received.
        WorkerEventTypeUnsupported: If an unsupported celery worker event was received.
        EventTypeUnknown: If an unknown event type (neither job nor worker) was received.
    """
    if event['type'].startswith('task'):
        factory = {
            JobEventName.Started: JobStartedEvent,
            JobEventName.Succeeded: JobSucceededEvent,
            JobEventName.Stopped: JobStoppedEvent,
            JobEventName.Aborted: JobAbortedEvent
        }
        if event['type'] in factory:
            return factory[event['type']].from_event(event)
        else:
            raise JobEventTypeUnsupported(
                'Unsupported event type {}'.format(event['type']))
    elif event['type'].startswith('worker'):
        raise WorkerEventTypeUnsupported(
            'Unsupported event type {}'.format(event['type']))
    else:
        raise EventTypeUnknown('Unknown event type {}'.format(event['type']))