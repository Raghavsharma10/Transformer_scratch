def normalize(vectors):
        """
        Normalize a matrix of row vectors to unit length.

        Contains a shortcut if there are no zero vectors in the matrix.
        If there are zero vectors, we do some indexing tricks to avoid
        dividing by 0.

        Parameters
        ----------
        vectors : np.array
            The vectors to normalize.

        Returns
        -------
        vectors : np.array
            The input vectors, normalized to unit length.

        """
        if np.ndim(vectors) == 1:
            norm = np.linalg.norm(vectors)
            if norm == 0:
                return np.zeros_like(vectors)
            return vectors / norm

        norm = np.linalg.norm(vectors, axis=1)

        if np.any(norm == 0):

            nonzero = norm > 0

            result = np.zeros_like(vectors)

            n = norm[nonzero]
            p = vectors[nonzero]
            result[nonzero] = p / n[:, None]

            return result
        else:
            return vectors / norm[:, None]