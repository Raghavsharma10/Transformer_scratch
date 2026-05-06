def describe_all(self, full=False):
        """Prints description information about all tables registered
        Args:
            full (bool): Also prints description of post processors.
        """
        for table in self.tabs:
            yield self.tabs[table]().describe(full)