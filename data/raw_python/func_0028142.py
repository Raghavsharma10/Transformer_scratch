def get_filetypes(self):
        # type: () -> List[str]
        """Return list of filetypes in your data

        Returns:
            List[str]: List of filetypes
        """
        if not self.is_requestable():
            return [resource.get_file_type() for resource in self.get_resources()]
        return self._get_stringlist_from_commastring('file_types')