def parse(self, xsl='text'):
        """
        TODO: add double-quoted string literals that allows for escaped double quotes
        https://gist.github.com/prathe/2439752 or
        http://www.metaltoad.com/blog/regex-quoted-string-escapable-quotes
        """

        try:
            cmd_text, option_text = xsl.split(None, 1)
        except ValueError:
            cmd_text = xsl
            option_text = ''

        try:
            context, cmd = cmd_text.strip().lower().split(':', 1)
        except ValueError:
            cmd = cmd_text.lower()
            context = None

        if not cmd in DEFAULT_CMD_TO_CONTEXT_MAPPING:
            raise ParseError("unknown command %s" % cmd)

        if context and not context in CONTEXTS:
            raise ParseError("unknown context %s" % context)

        self.context = context
        self.cmd = cmd
        self.text = None
        self.meta_commands = []
        self.options = {}

        try:
            if cmd in ('choose', 'text', 'meta'):
                raise ValueError()
            option_name, expr = option_text.split('=', 1)
            option_name = option_name.strip().lower()
            expr = unescape(expr).strip("'").strip('"').strip()
            self.options = {option_name: expr}

        except ValueError:
            text = unescape(option_text)

            if cmd == 'meta':
                for mc in filter(lambda c: c, map(lambda c: c.strip(), text.lower().split(';'))):
                    if mc in META_COMMANDS:
                        # store in stack order
                        self.meta_commands = [mc] + self.meta_commands
                    else:
                        raise ParseError("unknown meta command %s" % self.text)

            else:
                self.text = text