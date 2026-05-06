def escape(self, text, quote = True):
        """
        Escape special characters in HTML
        """
        if isinstance(text, bytes):
            return escape_b(text, quote)
        else:
            return escape(text, quote)