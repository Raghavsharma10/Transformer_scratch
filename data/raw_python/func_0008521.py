def nearest(self, type="VP"):
        """ Returns the nearest chunk in the sentence with the given type.
            This can be used (for example) to find adverbs and adjectives related to verbs,
            as in: "the cat is ravenous" => is what? => "ravenous".
        """
        candidate, d = None, len(self.sentence.chunks)
        if isinstance(self, PNPChunk):
            i = self.sentence.chunks.index(self.chunks[0])
        else:
            i = self.sentence.chunks.index(self)
        for j, chunk in enumerate(self.sentence.chunks):
            if chunk.type.startswith(type) and abs(i-j) < d:
                candidate, d = chunk, abs(i-j)
        return candidate