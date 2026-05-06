def write_usage(self, prog, args='', prefix='Usage: '):
        """Writes a usage line into the buffer.

        :param prog: the program name.
        :param args: whitespace separated list of arguments.
        :param prefix: the prefix for the first line.
        """
        prefix = '%*s%s' % (self.current_indent, prefix, prog)
        self.write(prefix)

        text_width = max(self.width - self.current_indent - term_len(prefix), 10)
        indent = ' ' * (term_len(prefix) + 1)
        self.write(wrap_text(args, text_width,
                             initial_indent=' ',
                             subsequent_indent=indent))

        self.write('\n')