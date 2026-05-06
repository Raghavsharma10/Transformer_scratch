def discard_event(event: events.Event, bot_id: str = None) -> bool:
    """
    Check if the incoming event needs to be discarded

    Args:
        event: Incoming :class:`slack.events.Event`
        bot_id: Id of connected bot

    Returns:
        boolean
    """
    if event["type"] in SKIP_EVENTS:
        return True
    elif bot_id and isinstance(event, events.Message):
        if event.get("bot_id") == bot_id:
            LOG.debug("Ignoring event: %s", event)
            return True
        elif "message" in event and event["message"].get("bot_id") == bot_id:
            LOG.debug("Ignoring event: %s", event)
            return True
    return False