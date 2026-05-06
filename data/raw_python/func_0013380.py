def removeRnaQuantificationSet(self, rnaQuantificationSet):
        """
        Removes the specified rnaQuantificationSet from this repository. This
        performs a cascading removal of all items within this
        rnaQuantificationSet.
        """
        q = models.Rnaquantificationset.delete().where(
            models.Rnaquantificationset.id == rnaQuantificationSet.getId())
        q.execute()