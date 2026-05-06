def process_header(self, headers):
        """Ignore the incomming header and replace it with the destination header"""

        return [c.name for c in self.source.dest_table.columns][1:]