def count(self, entity, files=False):
        """
        Return the count of unique values or files for the named entity.

        Args:
            entity (str): The name of the entity.
            files (bool): If True, counts the number of filenames that contain
                at least one value of the entity, rather than the number of
                unique values of the entity.
        """
        return self._find_entity(entity).count(files)