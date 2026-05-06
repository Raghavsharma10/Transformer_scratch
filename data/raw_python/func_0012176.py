def _batch(self,
               vectors,
               batch_size,
               num,
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
            if num == 1:
                sorted_indices = np.argmax(distances, 1)[:, None]
            else:
                sorted_indices = np.argpartition(-distances, kth=num, axis=1)
                sorted_indices = sorted_indices[:, :num]
            for lidx, indices in enumerate(sorted_indices):
                dists = distances[lidx, indices]
                if return_names:
                    dindex = np.argsort(-dists)
                    yield [(self.indices[indices[d]], dists[d])
                           for d in dindex]
                else:
                    yield list(-1 * np.sort(-dists))