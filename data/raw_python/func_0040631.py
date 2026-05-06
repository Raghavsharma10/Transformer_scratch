def makePartitions(self):
        """Make partitions with gmane help.
        """
        class NetworkMeasures:
            pass
        self.nm=nm=NetworkMeasures()
        nm.degrees=self.network.degree()
        nm.nodes_= sorted(self.network.nodes(), key=lambda x : nm.degrees[x])
        nm.degrees_=[nm.degrees[i] for i in nm.nodes_]
        nm.edges=     self.network.edges(data=True)
        nm.E=self.network.number_of_edges()
        nm.N=self.network.number_of_nodes()
        self.np=g.NetworkPartitioning(nm,10,metric="g")