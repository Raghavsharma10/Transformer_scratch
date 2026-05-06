def remove_note(self, username, index):
        """Remove a single usernote from the usernotes.

        Arguments:
            username: the user that for whom you're removing a note (str)
            index: the index of the note which is to be removed (int)

        Returns the update message for the usernotes wiki
        """
        self.cached_json['users'][username]['ns'].pop(index)

        # Go ahead and remove the user's entry if they have no more notes left
        if len(self.cached_json['users'][username]['ns']) == 0:
            del self.cached_json['users'][username]

        return '"delete note #{} on user {}" via puni'.format(index, username)