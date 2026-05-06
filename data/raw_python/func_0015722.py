def pprint(self, file_=sys.stdout):
        """Print the code block to stdout.
        Does syntax highlighting if possible.
        """

        code = []
        if self._deps:
            code.append("# dependencies:")
        for k, v in _compat.iteritems(self._deps):
            code.append("#   %s: %r" % (k, v))
        code.append(str(self))
        code = "\n".join(code)

        if file_.isatty():
            try:
                from pygments import highlight
                from pygments.lexers import PythonLexer
                from pygments.formatters import TerminalFormatter
            except ImportError:
                pass
            else:
                formatter = TerminalFormatter(bg="dark")
                lexer = PythonLexer()
                file_.write(highlight(code, lexer, formatter))
                return
        file_.write(code + "\n")