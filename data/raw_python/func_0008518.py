def related(self):
        """ Yields a list of all chunks in the sentence with the same relation id.
        """
        return [ch for ch in self.sentence.chunks 
                    if ch != self and intersects(unzip(0, ch.relations), unzip(0, self.relations))]