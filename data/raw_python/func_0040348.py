def prepare_notes(self, *notes, **keyword_notes):
        """Get injection values for all given notes."""
        __partial = keyword_notes.pop('__partial', False)
        args = tuple(self.get(note) for note in notes)
        kwargs = {}
        for arg in keyword_notes:
            note = keyword_notes[arg]
            if isinstance(note, tuple) and len(note) == 2 and note[0] == MAYBE:
                try:
                    kwargs[arg] = self.get(note[1])
                except LookupError:
                    continue
            elif __partial:
                try:
                    kwargs[arg] = self.get(note)
                except LookupError:
                    continue
            else:
                kwargs[arg] = self.get(note)
        return args, kwargs