def removeContinuousSet(self, continuousSet):
        """
        Removes the specified continuousSet from this repository.
        """
        q = models.ContinuousSet.delete().where(
            models.ContinuousSet.id == continuousSet.getId())
        q.execute()