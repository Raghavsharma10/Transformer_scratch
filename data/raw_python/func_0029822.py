def text(self):
        """Interpret the scalar as Markdown, strip the HTML and return text"""

        s = MLStripper()
        s.feed(self.html)
        return s.get_data()