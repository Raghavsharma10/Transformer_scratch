def submitbutton(self, request, tag):
        """
        Render an INPUT element of type SUBMIT which will post this form to the
        server.
        """
        return tags.input(type='submit',
                          name='__submit__',
                          value=self._getDescription())