def html(self):
        """Gives an html representation of the assessment."""
        output = self.html_preamble
        output += self._repr_html_()
        output += self.html_post
        return output