def has_better_coverage(self):
        """
        Return `True` if coverage of has improved, `False` otherwise.

        This does not ensure that all changes have been covered. If this is
        what you want, use `CoberturaDiff.has_all_changes_covered()` instead.
        """
        for filename in self.files():
            if self.diff_total_misses(filename) > 0:
                return False
        return True