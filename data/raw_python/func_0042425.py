def rows(self) -> List[List[str]]:
        """
        Returns the table rows.
        """
        return [list(d.values()) for d in self.data]