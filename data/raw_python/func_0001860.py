def publish(self, titles):
        """
        Publish a set of episodes to the Podcast's RSS feed.

        :param titles:
            Either a single episode title or a sequence of episode titles to
            publish.
        """
        if isinstance(titles, Sequence) and not isinstance(titles, six.string_types):
            for title in titles:
                self.episodes[title].publish()
        elif isinstance(titles, six.string_types):
            self.episodes[titles].publish()
        else:
            raise TypeError('titles must be a string or a sequence of strings.')

        self.update_rss_feed()