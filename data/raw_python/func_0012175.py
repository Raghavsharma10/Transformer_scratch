def _threshold_batch(self,
                         vectors,
                         batch_size,
                         threshold,
                         show_progressbar,
                         return_names):
        """Batched cosine distance."""
        vectors = self.normalize(vectors)

        # Single transpose, makes things faster.
        reference_transposed = self.norm_vectors.T

        for i in tqdm(range(0, len(vectors), batch_size),
                      disable=not show_progressbar):

            distances = vectors[i: i+batch_size].dot(reference_transposed)
            # For safety we clip
            distances = np.clip(distances, a_min=.0, a_max=1.0)
            for lidx, dists in enumerate(distances):
                indices = np.flatnonzero(dists >= threshold)
                sorted_indices = indices[np.argsort(-dists[indices])]
                if return_names:
                    yield [(self.indices[d], dists[d])
                           for d in sorted_indices]
                else:
                    yield list(dists[sorted_indices])