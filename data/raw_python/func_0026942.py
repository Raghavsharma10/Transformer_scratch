def parse_feeds(self, message_channel=True):
        """
        Iterates through each of the feed URLs, parses their items, and
        sends any items to the channel that have not been previously
        been parsed.
        """
        if parse:
            for feed_url in self.feeds:
                feed = parse(feed_url)
                for item in feed.entries:
                    if item["id"] not in self.feed_items:
                        self.feed_items.add(item["id"])
                        if message_channel:
                            message = self.format_item_message(feed, item)
                            self.message_channel(message)
                            return