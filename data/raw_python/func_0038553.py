def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (n_query, n_features).
            Test samples.

        Returns
        -------
        y : array of shape [n_samples]
            Class labels for each data sample.

        """
        # TODO: Make classification of multiple samples a bit more effective...
        if X.ndim > 1 and X.shape[1] != 1:
            out = []
            for x in X:
                out += self.predict(x)
            return out
        X = X.flatten()

        if self.metric == 'minkowski':
            dists = np.sum(np.abs(self._data - X) ** self.p, axis=1)
        else:
            # TODO: Implement other metrics.
            raise ValueError("Only Minkowski distance metric implemented...")

        argument = np.argsort(dists)
        labels = self._labels[argument[:self.n_neighbors]]

        if self.weights == 'distance':
            weights = 1 / dists[argument[:self.n_neighbors]]
            out = np.zeros((len(self._classes), ), 'float')
            for i, c in enumerate(self._classes):
               out[i] = np.sum(weights[labels == c])
            out /= np.sum(out)
            y_pred = self._labels[np.argmax(out)]
        else:
            y_pred, _ = mode(labels)

        return y_pred.tolist()