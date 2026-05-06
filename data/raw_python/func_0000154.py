def get_notes(self, user):
        """Return a list of Note objects for the given user.

        Return an empty list if no notes are found.

        Arguments:
            user: the user to search for in the usernotes (str)
        """
        # Try to search for all notes on a user, return an empty list if none
        # are found.
        try:
            users_notes = []

            for note in self.cached_json['users'][user]['ns']:
                users_notes.append(Note(
                    user=user,
                    note=note['n'],
                    subreddit=self.subreddit,
                    mod=self._mod_from_index(note['m']),
                    link=note['l'],
                    warning=self._warning_from_index(note['w']),
                    note_time=note['t']
                ))

            return users_notes
        except KeyError:
            # User not found
            return []