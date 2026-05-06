def mark_seen(self, medium):
        """
        Creates EventSeen objects for the provided medium for every event
        in the queryset.

        Creating these EventSeen objects ensures they will not be
        returned when passing ``seen=False`` to any of the medium
        event retrieval functions, ``events``, ``entity_events``, or
        ``events_targets``.
        """
        EventSeen.objects.bulk_create([
            EventSeen(event=event, medium=medium) for event in self
        ])