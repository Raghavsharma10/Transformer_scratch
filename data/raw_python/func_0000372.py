def format_choices(self):
        """Return the choices in string form."""
        ce = enumerate(self.choices)
        f = lambda i, c: '%s (%d)' % (c, i+1)
        # apply formatter and append help token
        toks = [f(i,c) for i, c in ce] + ['Help (?)']
        return ' '.join(toks)