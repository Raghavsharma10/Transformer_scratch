def parse_note(cls, note):
        """Parse string annotation into object reference with optional name."""
        if isinstance(note, tuple):
            if len(note) != 2:
                raise ValueError('tuple annotations must be length 2')
            return note
        try:
            match = cls.re_note.match(note)
        except TypeError:
            # Note is not a string. Support any Python object as a note.
            return note, None
        return match.groups()