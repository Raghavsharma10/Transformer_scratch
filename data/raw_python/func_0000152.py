def _init_notes(self):
        """Set up the UserNotes page with the initial JSON schema."""
        self.cached_json = {
            'ver': self.schema,
            'users': {},
            'constants': {
                'users': [x.name for x in self.subreddit.moderator()],
                'warnings': Note.warnings
            }
        }

        self.set_json('Initializing JSON via puni', True)