def delete(self):
        """Deletes the current curve.

        :raises RuntimeError: Raises when` when one tries to delete a read-only
            curve.

        """
        if self._writeable:
            self._write(('CRVDEL', Integer), self.idx)
        else:
            raise RuntimeError('Can not delete read-only curves.')