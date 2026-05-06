def reset_mock(self, visited=None):
        """Reset the default tell/read/write/etc side effects."""
        # In some versions of the mock library, `reset_mock` takes an argument
        # and in some it doesn't. We try to handle all situations.
        if visited is not None:
            super(FileLikeMock, self).reset_mock(visited)
        else:
            super(FileLikeMock, self).reset_mock()

        # Reset contents and tell/read/write/close side effects.
        self.read_data = ''
        self.close.side_effect = self._close