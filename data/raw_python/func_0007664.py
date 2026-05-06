def get_stats(self, i):
        """Gets the standard statistics for aux_index `i`. For example, if `token_generator` generates
        `(text_idx, sentence_idx, word)`, then `get_stats(0)` will return various statistics about sentence lengths
        across texts. Similarly, `get_counts(1)` will return statistics of token lengths across sentences.

        This information can be used to pad or truncate inputs.
        """
        # OrderedDict to always show same order if printed.
        result = OrderedDict()
        result['min'] = np.min(self._counts[i])
        result['max'] = np.max(self._counts[i])
        result['std'] = np.std(self._counts[i])
        result['mean'] = np.mean(self._counts[i])
        return result