def set_json(self, reason='', new_page=False):
        """Send the JSON from the cache to the usernotes wiki page.

        Arguments:
            reason: the change reason that will be posted to the wiki changelog
                (str)
        Raises:
            OverflowError if the new JSON data is greater than max_page_size
        """
        compressed_json = json.dumps(self._compress_json(self.cached_json))

        if len(compressed_json) > self.max_page_size:
            raise OverflowError(
                'Usernotes page is too large (>{0} characters)'.
                format(self.max_page_size)
            )

        if new_page:
            self.subreddit.wiki.create(
                self.page_name,
                compressed_json,
                reason
            )
            # Set the page as hidden and available to moderators only
            self.subreddit.wiki[self.page_name].mod.update(False, permlevel=2)
        else:
            self.subreddit.wiki[self.page_name].edit(
                compressed_json,
                reason
            )