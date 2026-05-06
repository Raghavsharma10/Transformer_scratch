def print_out(self, value, indent=None, format_options=None, asap=False):
        """Prints out the given value.

        :param value:

        :param str|unicode indent:

        :param dict|str|unicode format_options: text color

        :param bool asap: Print as soon as possible.

        """
        if indent is None:
            indent = '>   '

        text = indent + str(value)

        if format_options is None:
            format_options = 'gray'

        if self._style_prints and format_options:

            if not isinstance(format_options, dict):
                format_options = {'color_fg': format_options}

            text = format_print_text(text, **format_options)

        command = 'iprint' if asap else 'print'
        self._set(command, text, multi=True)

        return self