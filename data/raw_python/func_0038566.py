def read_sif(cls, path):
        """
        Creates a graph from a `simple interaction format (SIF)`_ file

        Parameters
        ----------
        path : str
            Absolute path to a SIF file

        Returns
        -------
        caspo.core.graph.Graph
            Created object instance


        .. _simple interaction format (SIF): http://wiki.cytoscape.org/Cytoscape_User_Manual/Network_Formats
        """
        df = pd.read_csv(path, delim_whitespace=True, names=['source', 'sign', 'target']).drop_duplicates()
        edges = [(source, target, {'sign': sign}) for _, source, sign, target in df.itertuples()]
        return cls(data=edges)