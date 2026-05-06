def _check(self, terms):
        """Check terms do not contain unknown characters"""
        for t in terms:
            try:
                _ = urllib.parse.quote(six.text_type(t).encode('utf8'))
            except:
                self.logger.error('Unknown character in [{0}]!'.format(t))
                self.logger.error('.... remove character and try again.')
                raise EncodingError