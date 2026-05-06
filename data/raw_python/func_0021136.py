def get_event_counts(self):
        """
        Returns a dict like:
            {'counts': {
                'all': 30,
                'movie': 12,
                'gig': 10,
            }}
        """
        counts = {'all': Event.objects.count(),}

        for k,v in Event.KIND_CHOICES:
            # e.g. 'movie_count':
            counts[k] = Event.objects.filter(kind=k).count()

        return {'counts': counts,}