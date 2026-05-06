def has_all_changes_covered(self):
        """
        Return `True` if all changes have been covered, `False` otherwise.
        """
        for filename in self.files():
            for hunk in self.file_source_hunks(filename):
                for line in hunk:
                    if line.reason is None:
                        continue  # line untouched
                    if line.status is False:
                        return False  # line not covered
        return True