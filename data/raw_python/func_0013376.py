def insertBiosample(self, biosample):
        """
        Inserts the specified Biosample into this repository.
        """
        try:
            models.Biosample.create(
                id=biosample.getId(),
                datasetid=biosample.getParentContainer().getId(),
                name=biosample.getLocalId(),
                description=biosample.getDescription(),
                disease=json.dumps(biosample.getDisease()),
                created=biosample.getCreated(),
                updated=biosample.getUpdated(),
                individualid=biosample.getIndividualId(),
                attributes=json.dumps(biosample.getAttributes()),
                individualAgeAtCollection=json.dumps(
                        biosample.getIndividualAgeAtCollection()))
        except Exception:
            raise exceptions.DuplicateNameException(
                biosample.getLocalId(),
                biosample.getParentContainer().getLocalId())