def close(self):
        """Close file, see file.close"""
        try:
            self.parent_fd.fileno()
        except io.UnsupportedOperation:
            logger.debug("Not closing parent_fd - reusing existing")
        else:
            self.parent_fd.close()