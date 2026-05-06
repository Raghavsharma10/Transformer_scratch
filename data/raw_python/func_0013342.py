def allReadGroups(self):
        """
        Return an iterator over all read groups in the data repo
        """
        for dataset in self.getDatasets():
            for readGroupSet in dataset.getReadGroupSets():
                for readGroup in readGroupSet.getReadGroups():
                    yield readGroup