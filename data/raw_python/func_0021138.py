def get_queryset(self):
        "Restrict to a single kind of event, if any, and include Venue data."
        qs = super().get_queryset()

        kind = self.get_event_kind()
        if kind is not None:
            qs = qs.filter(kind=kind)

        qs = qs.select_related('venue')

        return qs