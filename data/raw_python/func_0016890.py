def undelete(self, original_filepath):
        """Restore the most recent version of a filepath, returning
        the filepath it was restored to(as rename-on-collision will
        apply if a file already exists at that path).
        """
        candidates = self.versions(original_filepath)
        if not candidates:
            raise x_not_found_in_recycle_bin("%s not found in the Recycle Bin" % original_filepath)
        #
        # NB Can't use max(key=...) until Python 2.6+
        #
        newest = sorted(candidates, key=lambda entry: entry.recycle_date())[-1]
        return newest.undelete()