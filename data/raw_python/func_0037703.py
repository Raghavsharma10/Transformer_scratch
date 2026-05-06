def sponsored(self, **kwargs):
        """Search containing any sponsored pieces of Content."""
        eqs = self.search(**kwargs)
        eqs = eqs.filter(AllSponsored())
        published_offset = getattr(settings, "RECENT_SPONSORED_OFFSET_HOURS", None)
        if published_offset:
            now = timezone.now()
            eqs = eqs.filter(
                Published(
                    after=now - timezone.timedelta(hours=published_offset),
                    before=now
                )
            )
        return eqs