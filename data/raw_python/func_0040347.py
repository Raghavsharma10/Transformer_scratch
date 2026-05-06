def prepare_callable(self, fn, partial=False):
        """Prepare arguments required to apply function."""
        notes, keyword_notes = self.get_annotations(fn)
        return self.prepare_notes(*notes, __partial=partial, **keyword_notes)