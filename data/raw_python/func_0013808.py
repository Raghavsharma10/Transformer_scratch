def stats(self, start_date=None, end_date=None, registered_only=False):
        """Returns a dictionary of pageviews including:

            * total pageviews

        for all users, registered users and guests.
        """
        pageviews = self.filter(
            visitor__start_time__lt=end_date,
            visitor__start_time__gte=start_date,
        ).select_related('visitor')

        stats = {
            'total': 0,
            'unique': 0,
        }

        stats['total'] = total_views = pageviews.count()
        unique_count = 0

        if not total_views:
            return stats

        # Registered user sessions
        registered_pageviews = pageviews.filter(visitor__user__isnull=False)
        registered_count = registered_pageviews.count()

        if registered_count:
            registered_unique_count = registered_pageviews.values(
                'visitor', 'url').distinct().count()

            # Update the total unique count...
            unique_count += registered_unique_count

            stats['registered'] = {
                'total': registered_count,
                'unique': registered_unique_count,
            }

        if TRACK_ANONYMOUS_USERS and not registered_only:
            guest_pageviews = pageviews.filter(visitor__user__isnull=True)
            guest_count = guest_pageviews.count()

            if guest_count:
                guest_unique_count = guest_pageviews.values(
                    'visitor', 'url').distinct().count()

                # Update the total unique count...
                unique_count += guest_unique_count

                stats['guests'] = {
                    'total': guest_count,
                    'unique': guest_unique_count,
                }

        # Finish setting the total visitor counts
        stats['unique'] = unique_count

        return stats