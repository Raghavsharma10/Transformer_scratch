def update_spec(self):
        """Update the source specification with information from the row intuiter, but only if the spec values
        are not already set. """

        if self.datafile.exists:
            with self.datafile.reader as r:

                self.header_lines = r.info['header_rows']
                self.comment_lines = r.info['comment_rows']
                self.start_line = r.info['data_start_row']
                self.end_line = r.info['data_end_row']