def summarize_dataframe(self):
        """Summarize default dataframe for this cohort using a hash function.
        Useful for confirming the version of data used in various reports, e.g. ipynbs
        """
        if self.dataframe_hash:
            return(self.dataframe_hash)
        else:
            df = self._as_dataframe_unmodified()
            return(self.dataframe_hash)