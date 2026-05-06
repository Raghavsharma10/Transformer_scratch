def traces(self):
        """
        Decomposes the Chromatogram into a collection of Traces.

        Returns
        -------
        list
        """
        traces = []
        for v, c in zip(self.values.T, self.columns):
            traces.append(Trace(v, self.index, name=c))
        return traces