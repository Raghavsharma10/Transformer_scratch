def finish(self):
        """
        Parse the buffered response body, rewrite its URLs, write the result to
        the wrapped request, and finish the wrapped request.
        """
        stylesheet = ''.join(self._buffer)
        parser = CSSParser()
        css = parser.parseString(stylesheet)
        replaceUrls(css, self._replace)
        self.request.write(css.cssText)
        return self.request.finish()