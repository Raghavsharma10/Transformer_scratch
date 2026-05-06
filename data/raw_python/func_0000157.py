def add_note(self, note):
        """Add a note to the usernotes wiki page.

        Arguments:
            note: the note to be added (Note)

        Returns the update message for the usernotes wiki

        Raises:
            ValueError when the warning type of the note can not be found in the
                stored list of warnings.
        """
        notes = self.cached_json

        if not note.moderator:
            note.moderator = self.r.user.me().name

        # Get index of moderator in mod list from usernotes
        # Add moderator to list if not already there
        try:
            mod_index = notes['constants']['users'].index(note.moderator)
        except ValueError:
            notes['constants']['users'].append(note.moderator)
            mod_index = notes['constants']['users'].index(note.moderator)

        # Get index of warning type from warnings list
        # Add warning type to list if not already there
        try:
            warn_index = notes['constants']['warnings'].index(note.warning)
        except ValueError:
            if note.warning in Note.warnings:
                notes['constants']['warnings'].append(note.warning)
                warn_index = notes['constants']['warnings'].index(note.warning)
            else:
                raise ValueError('Warning type not valid: ' + note.warning)

        new_note = {
            'n': note.note,
            't': note.time,
            'm': mod_index,
            'l': note.link,
            'w': warn_index
        }

        try:
            notes['users'][note.username]['ns'].insert(0, new_note)
        except KeyError:
            notes['users'][note.username] = {'ns': [new_note]}

        return '"create new note on user {}" via puni'.format(note.username)