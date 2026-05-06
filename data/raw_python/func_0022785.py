def _get_error(self, code, errors, indentation=0):
        """Get error and show the faulty line + some context
        Other GLIR implementations may omit this.
        """
        # Init
        results = []
        lines = None
        if code is not None:
            lines = [line.strip() for line in code.split('\n')]

        for error in errors.split('\n'):
            # Strip; skip empy lines
            error = error.strip()
            if not error:
                continue
            # Separate line number from description (if we can)
            linenr, error = self._parse_error(error)
            if None in (linenr, lines):
                results.append('%s' % error)
            else:
                results.append('on line %i: %s' % (linenr, error))
                if linenr > 0 and linenr < len(lines):
                    results.append('  %s' % lines[linenr - 1])

        # Add indentation and return
        results = [' ' * indentation + r for r in results]
        return '\n'.join(results)