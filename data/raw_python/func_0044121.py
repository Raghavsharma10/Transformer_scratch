def summarize_events():
    """Some information about active events and callbacks."""

    for ev in event.events:
        if ev.callbacks:
            LOG.info("subscribed to %s by %s", ev, ', '.join(imap(repr, ev.callbacks)))