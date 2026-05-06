def add(self, *args, **kwargs):
        """Add a new entry to the feed.  This function can either be called
        with a :class:`FeedEntry` or some keyword and positional arguments
        that are forwarded to the :class:`FeedEntry` constructor.
        """
        if len(args) == 1 and not kwargs and isinstance(args[0], FeedEntry):
            self.entries.append(args[0])
        else:
            kwargs['feed_url'] = self.feed_url
            kwargs['timezone'] = self.timezone
            self.entries.append(FeedEntry(*args, **kwargs))