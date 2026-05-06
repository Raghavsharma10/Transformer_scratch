def addBiosample(self, biosample):
        """
        Adds the specified biosample to this dataset.
        """
        id_ = biosample.getId()
        self._biosampleIdMap[id_] = biosample
        self._biosampleIds.append(id_)
        self._biosampleNameMap[biosample.getName()] = biosample