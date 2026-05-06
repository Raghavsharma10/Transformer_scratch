def get_binary_iterator(self):
        """
        Generator to stream the remote file piece by piece.
        """
        CHUNK_SIZE = 1024
        return (item for item in requests.get(self.url).iter_content(CHUNK_SIZE))