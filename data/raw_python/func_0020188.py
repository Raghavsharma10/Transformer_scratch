def _escape(self, text):
        """Escape text according to self.escape"""
        ret = EMPTYSTRING if text is None else str(text)
        if self.escape:
            return html_escape(ret)
        else:
            return ret