def delete(self, name):
        """
        Deletes the named file.
        :param name: the name.
        :return: 200 if it was deleted, 404 if it doesn't exist or 500 for anything else.
        """
        try:
            result = self._uploadController.delete(name)
            return None, 200 if result is not None else 404
        except Exception as e:
            return str(e), 500