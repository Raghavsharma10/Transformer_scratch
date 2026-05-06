def removeIndividual(self, individual):
        """
        Removes the specified individual from this repository.
        """
        q = models.Individual.delete().where(
            models.Individual.id == individual.getId())
        q.execute()