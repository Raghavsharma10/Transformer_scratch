def insertIndividual(self, individual):
        """
        Inserts the specified individual into this repository.
        """
        try:
            models.Individual.create(
                id=individual.getId(),
                datasetId=individual.getParentContainer().getId(),
                name=individual.getLocalId(),
                description=individual.getDescription(),
                created=individual.getCreated(),
                updated=individual.getUpdated(),
                species=json.dumps(individual.getSpecies()),
                sex=json.dumps(individual.getSex()),
                attributes=json.dumps(individual.getAttributes()))
        except Exception:
            raise exceptions.DuplicateNameException(
                individual.getLocalId(),
                individual.getParentContainer().getLocalId())