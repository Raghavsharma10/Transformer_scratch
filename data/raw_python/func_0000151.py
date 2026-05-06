def get_json(self):
        """Get the JSON stored on the usernotes wiki page.

        Returns a dict representation of the usernotes (with the notes BLOB
        decoded).

        Raises:
            RuntimeError if the usernotes version is incompatible with this
                version of puni.
        """
        try:
            usernotes = self.subreddit.wiki[self.page_name].content_md
            notes = json.loads(usernotes)
        except NotFound:
            self._init_notes()
        else:
            if notes['ver'] != self.schema:
                raise RuntimeError(
                    'Usernotes schema is v{0}, puni requires v{1}'.
                    format(notes['ver'], self.schema)
                )

            self.cached_json = self._expand_json(notes)

        return self.cached_json