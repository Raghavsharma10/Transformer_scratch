def _build_highlight(self, fields, options):
        """Return the portion of the query that controls highlighting."""
        ret = {'fields': dict((f, {}) for f in fields),
               'order': 'score'}
        ret.update(options)
        return ret