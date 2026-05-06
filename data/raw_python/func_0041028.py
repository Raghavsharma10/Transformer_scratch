def _dash_escape_text(text: str) -> str:
        """
        Add dash '-' (0x2D) and space ' ' (0x20) as prefix on each line

        :param text: Text to dash-escape
        :return:
        """
        dash_escaped_text = str()

        for line in text.splitlines(True):
            # add dash '-' (0x2D) and space ' ' (0x20) as prefix
            dash_escaped_text += DASH_ESCAPE_PREFIX + line

        return dash_escaped_text