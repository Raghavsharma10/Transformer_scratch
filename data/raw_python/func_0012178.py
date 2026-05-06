def vector_similarity(self, vector, items):
        """Compute the similarity between a vector and a set of items."""
        vector = self.normalize(vector)
        items_vec = np.stack([self.norm_vectors[self.items[x]] for x in items])
        return vector.dot(items_vec.T)