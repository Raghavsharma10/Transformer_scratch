def calculate_checksum(self):
        """Calculate ISBN checksum.

        Returns:
            ``str``: ISBN checksum value

        """
        if len(self.isbn) in (9, 12):
            return calculate_checksum(self.isbn)
        else:
            return calculate_checksum(self.isbn[:-1])