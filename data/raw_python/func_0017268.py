def distance(self, word1, word2, metric='levenshtein'):
        """Get the similarity of the words, using the supported distance metrics."""
        if metric in self.distances:
            distance_func = self.distances[metric]
            return distance_func(self.phonetics(word1), self.phonetics(word2))
        else:
            raise DistanceMetricError('Distance metric not supported! Choose from levenshtein, hamming.')