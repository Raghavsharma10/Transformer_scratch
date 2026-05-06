def get_queryset(self):
        "Reduce the number of queries and speed things up."
        qs = super().get_queryset()

        qs = qs.select_related('publication__series') \
                .prefetch_related('publication__roles__creator')

        return qs