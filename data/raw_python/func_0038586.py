def save(self, path):
        """
        Writes file to a particular location

        This won't work for cloud environments like Google's App Engine, use with caution
        ensure to catch exceptions so you can provide informed feedback.

        prestans does not mask File IO exceptions so your handler can respond better.
        """

        file_handle = open(path, 'wb')
        file_handle.write(self._file_contents)
        file_handle.close()