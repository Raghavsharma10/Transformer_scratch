def removeBiosample(self, biosample):
        """
        Removes the specified biosample from this repository.
        """
        q = models.Biosample.delete().where(
            models.Biosample.id == biosample.getId())
        q.execute()