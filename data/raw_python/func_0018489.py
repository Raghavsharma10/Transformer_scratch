def mirror(self, tables, dest_url):
        """
        miror a set of `tables` from `dest_url`

        Returns a new Genome object

        Parameters
        ----------

        tables : list
            an iterable of tables

        dest_url: str
            a dburl string, e.g. 'sqlite:///local.db'
        """
        from mirror import mirror
        return mirror(self, tables, dest_url)