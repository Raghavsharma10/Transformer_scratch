def cleanlines(self):
        """Cleaned code lines.

        Implemented cleanups:
          * eventually remove method version
          * remove docstrings
          * remove comments
          * remove empty lines
          * remove line brackes within brackets
          * replace `modelutils` with nothing
          * remove complete lines containing `fastaccess`
          * replace shortcuts with complete references
        """
        code = inspect.getsource(self.func)
        code = '\n'.join(code.split('"""')[::2])
        code = code.replace('modelutils.', '')
        for (name, shortcut) in zip(self.collectornames,
                                    self.collectorshortcuts):
            code = code.replace('%s.' % shortcut, 'self.%s.' % name)
        code = self.remove_linebreaks_within_equations(code)
        lines = code.splitlines()
        self.remove_imath_operators(lines)
        lines[0] = 'def %s(self):' % self.funcname
        lines = [l.split('#')[0] for l in lines]
        lines = [l for l in lines if 'fastaccess' not in l]
        lines = [l.rstrip() for l in lines if l.rstrip()]
        return Lines(*lines)