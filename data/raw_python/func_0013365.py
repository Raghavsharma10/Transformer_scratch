def removeReadGroupSet(self, readGroupSet):
        """
        Removes the specified readGroupSet from this repository. This performs
        a cascading removal of all items within this readGroupSet.
        """
        for readGroupSetRecord in models.Readgroupset.select().where(
                        models.Readgroupset.id == readGroupSet.getId()):
            readGroupSetRecord.delete_instance(recursive=True)