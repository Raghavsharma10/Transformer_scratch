def close(self):
        """Close the stream."""
        self.flush()
        self.stream.close()
        logging.StreamHandler.close(self)