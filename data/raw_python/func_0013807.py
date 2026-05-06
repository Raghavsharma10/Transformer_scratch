def stats(self, start_date, end_date, registered_only=False):
        """Returns a dictionary of visits including:

            * total visits
            * unique visits
            * return ratio
            * pages per visit (if pageviews are enabled)
            * time on site

        for all users, registered users and guests.
        """
        visitors = self.filter(
            start_time__gte=start_date,
            start_time__lt=end_date
        )

        stats = {
            'total': 0,
            'unique': 0,
            'return_ratio': 0,
        }

        # All visitors
        stats['total'] = total_count = visitors.count()
        unique_count = 0

        # No visitors! Nothing more to do.
        if not total_count:
            return stats

        # Avg time on site
        total_time_on_site = visitors.aggregate(
            avg_tos=Avg('time_on_site'))['avg_tos']
        stats['time_on_site'] = timedelta(seconds=int(total_time_on_site))

        # Registered user sessions
        registered_visitors = visitors.filter(user__isnull=False)
        registered_total_count = registered_visitors.count()

        if registered_total_count:
            registered_unique_count = registered_visitors.values(
                'user'
            ).distinct().count()
            # Avg time on site
            time_on_site = registered_visitors.aggregate(
                avg_tos=Avg('time_on_site'))['avg_tos']

            # Update the total unique count..
            unique_count += registered_unique_count

            # Set the registered stats..
            returns = (registered_total_count - registered_unique_count)
            stats['registered'] = {
                'total': registered_total_count,
                'unique': registered_unique_count,
                'return_ratio': (returns / registered_total_count) * 100,
                'time_on_site': timedelta(seconds=int(time_on_site)),
            }

        # Get stats for our guests..
        if TRACK_ANONYMOUS_USERS and not registered_only:
            guests = visitors.filter(user__isnull=True)
            guest_total_count = guests.count()

            if guest_total_count:
                guest_unique_count = guests.values(
                    'ip_address'
                ).distinct().count()
                # Avg time on site
                guest_time_on_site = guests.aggregate(
                    avg_tos=Avg('time_on_site'))['avg_tos']
                # return rate
                returns = (guest_total_count - guest_unique_count)
                return_ratio = (returns / guest_total_count) * 100
                time_on_site = timedelta(seconds=int(guest_time_on_site))
            else:
                guest_total_count = 0
                guest_unique_count = 0
                return_ratio = 0.0
                time_on_site = timedelta(0)

            # Update the total unique count
            unique_count += guest_unique_count
            stats['guests'] = {
                'total': guest_total_count,
                'unique': guest_unique_count,
                'return_ratio': return_ratio,
                'time_on_site': time_on_site,
            }

        # Finish setting the total visitor counts
        returns = (total_count - unique_count)
        stats['unique'] = unique_count
        stats['return_ratio'] = (returns / total_count) * 100

        # If pageviews are being tracked, add the aggregate pages-per-visit
        if TRACK_PAGEVIEWS:
            if 'registered' in stats:
                pages_per_visit = registered_visitors.annotate(
                    page_count=Count('pageviews')
                ).filter(page_count__gt=0).aggregate(
                    pages_per_visit=Avg('page_count'))['pages_per_visit']
                stats['registered']['pages_per_visit'] = pages_per_visit

            if TRACK_ANONYMOUS_USERS and not registered_only:
                stats['guests']['pages_per_visit'] = guests.annotate(
                    page_count=Count('pageviews')
                ).filter(page_count__gt=0).aggregate(
                    pages_per_visit=Avg('page_count'))['pages_per_visit']

                total_per_visit = visitors.annotate(
                    page_count=Count('pageviews')
                ).filter(page_count__gt=0).aggregate(
                    pages_per_visit=Avg('page_count'))['pages_per_visit']
            else:
                if 'registered' in stats:
                    total_per_visit = stats['registered']['pages_per_visit']
                else:
                    total_per_visit = 0

            stats['pages_per_visit'] = total_per_visit

        return stats