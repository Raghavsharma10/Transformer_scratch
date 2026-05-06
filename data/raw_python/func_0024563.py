def histogram(self, cycle_data, bins=10):
        """Return histogram data for the cycle times in `cycle_data`. Returns
        a dictionary with keys `bin_values` and `bin_edges` of numpy arrays
        """
        values, edges = np.histogram(cycle_data['cycle_time'].astype('timedelta64[D]').dropna(), bins=bins)

        index = []
        for i, v in enumerate(edges):
            if i == 0:
                continue
            index.append("%.01f to %.01f" % (edges[i - 1], edges[i],))

        return pd.Series(values, name="Items", index=index)