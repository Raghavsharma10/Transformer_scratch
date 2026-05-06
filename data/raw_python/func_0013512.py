def addIndividual(self, individual):
        """
        Adds the specified individual to this dataset.
        """
        id_ = individual.getId()
        self._individualIdMap[id_] = individual
        self._individualIds.append(id_)
        self._individualNameMap[individual.getName()] = individual