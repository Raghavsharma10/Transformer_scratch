def removeReferenceSet(self, referenceSet):
        """
        Removes the specified referenceSet from this repository. This performs
        a cascading removal of all references within this referenceSet.
        However, it does not remove any of the ReadGroupSets or items that
        refer to this ReferenceSet. These must be deleted before the
        referenceSet can be removed.
        """
        try:
            q = models.Reference.delete().where(
                    models.Reference.referencesetid == referenceSet.getId())
            q.execute()
            q = models.Referenceset.delete().where(
                    models.Referenceset.id == referenceSet.getId())
            q.execute()
        except Exception:
            msg = ("Unable to delete reference set.  "
                   "There are objects currently in the registry which are "
                   "aligned against it.  Remove these objects before removing "
                   "the reference set.")
            raise exceptions.RepoManagerException(msg)