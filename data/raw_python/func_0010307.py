def write_file(self, *args, **kwargs):
        """Write a file into this directory

        This method takes the same arguments as :meth:`.FileDataAPI.write_file`
        with the exception of the ``path`` argument which is not needed here.

        """
        return self._fdapi.write_file(self.get_path(), *args, **kwargs)