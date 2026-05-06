def get_training_coverage(self):
        """
        Returns a ratio of classifiers that were able to be trained successfully.
        """
        total = len(self.training_results)
        i = sum(1 for data in self.training_results.values() if not isinstance(data, basestring))
        return i/float(total)