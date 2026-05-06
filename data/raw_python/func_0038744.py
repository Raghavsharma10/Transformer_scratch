def append(self, network):
        """
        Append a :class:`caspo.core.logicalnetwork.LogicalNetwork` to the list

        Parameters
        ----------
        network : :class:`caspo.core.logicalnetwork.LogicalNetwork`
            The network to append
        """
        arr = network.to_array(self.hg.mappings)
        if len(self.__matrix):
            self.__matrix = np.append(self.__matrix, [arr], axis=0)
            self.__networks = np.append(self.__networks, network.networks)
        else:
            self.__matrix = np.array([arr])
            self.__networks = np.array([network.networks])