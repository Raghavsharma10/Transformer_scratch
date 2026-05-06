def analysis(self):
        """Return an AnalysisPartition proxy, which wraps this partition to provide acess to
        dataframes, shapely shapes and other analysis services"""
        if isinstance(self, PartitionProxy):
            return AnalysisPartition(self._obj)
        else:
            return AnalysisPartition(self)