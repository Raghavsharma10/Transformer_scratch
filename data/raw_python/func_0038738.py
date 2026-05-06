def from_csv(cls, filename):
        """
        Creates a list of logical networks from a CSV file.
        Columns that cannot be parsed as a :class:`caspo.core.mapping.Mapping` are ignored
        except for a column named `networks` which (if present) is interpreted as the number
        of logical networks having the same input-output behavior.

        Parameters
        ----------
        filename : str
           Absolute path to CSV file

        Returns
        -------
        caspo.core.logicalnetwork.LogicalNetworkList
           Created object instance
        """
        df = pd.read_csv(filename)

        edges = set()
        mappings = []
        cols = []
        for m in df.columns:
            try:
                ct = Mapping.from_str(m)
                mappings.append(ct)
                cols.append(m)
                for source, sign in ct.clause:
                    edges.add((source, ct.target, sign))
            except ValueError:
                #current column isn't a mapping
                pass

        graph = Graph.from_tuples(edges)
        hypergraph = HyperGraph.from_graph(graph)
        hypergraph.mappings = mappings

        if 'networks' in df.columns:
            nnet = df['networks'].values.astype(int)
        else:
            nnet = None

        return cls(hypergraph, matrix=df[cols].values, networks=nnet)