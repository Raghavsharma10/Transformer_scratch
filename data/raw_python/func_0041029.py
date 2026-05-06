def _parse_dash_escaped_line(dash_escaped_line: str) -> str:
        """
        Parse a dash-escaped text line

        :param dash_escaped_line: Dash escaped text line
        :return:
        """
        text = str()
        regex_dash_escape_prefix = compile('^' + DASH_ESCAPE_PREFIX)
        # if prefixed by a dash escape prefix...
        if regex_dash_escape_prefix.match(dash_escaped_line):
            # remove dash '-' (0x2D) and space ' ' (0x20) prefix
            text += dash_escaped_line[2:]

        return text