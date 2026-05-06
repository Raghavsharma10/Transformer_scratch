def _cmp_key(self, obj=None):
        """Comparison key for sorting results from all linters.

        The sort should group files and lines from different linters to make it
        easier for refactoring.
        """
        if not obj:
            obj = self
        line_nr = int(obj.line_nr) if obj.line_nr else 0
        col = int(obj.col) if obj.col else 0
        return (obj.path, line_nr, col, obj.msg)