def nl2br(self, text):
        """
        Replace \'\n\' with \'<br/>\\n\'
        """
        if isinstance(text, bytes):
            return text.replace(b'\n', b'<br/>\n')
        else:
            return text.replace('\n', '<br/>\n')