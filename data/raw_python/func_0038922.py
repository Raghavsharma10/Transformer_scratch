def handle_set_row(self):
        """Read incoming row change from server"""
        row = self.reader.int()
        logger.info(" -> row: %s", row)
        self.controller.row = row