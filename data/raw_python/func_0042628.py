def _unseen_event_ids(medium):
    """
    Return all events that have not been seen on this medium.
    """
    query = '''
    SELECT event.id
    FROM entity_event_event AS event
        LEFT OUTER JOIN (SELECT *
                         FROM entity_event_eventseen AS seen
                         WHERE seen.medium_id=%s) AS eventseen
            ON event.id = eventseen.event_id
    WHERE eventseen.medium_id IS NULL
    '''
    unseen_events = Event.objects.raw(query, params=[medium.id])
    ids = [e.id for e in unseen_events]
    return ids