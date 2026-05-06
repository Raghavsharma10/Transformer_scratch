def _get_child_mock(self, **kws):
        """Create a new FileLikeMock instance.

        The new mock will inherit the parent's side_effect and read_data
        attributes.
        """
        kws.update({
            '_new_parent': self,
            'side_effect': self._mock_side_effect,
            'read_data': self.__read_data,
        })
        return FileLikeMock(**kws)