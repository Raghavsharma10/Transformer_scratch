def insertOntology(self, ontology):
        """
        Inserts the specified ontology into this repository.
        """
        try:
            models.Ontology.create(
                    id=ontology.getName(),
                    name=ontology.getName(),
                    dataurl=ontology.getDataUrl(),
                    ontologyprefix=ontology.getOntologyPrefix())
        except Exception:
            raise exceptions.DuplicateNameException(
                ontology.getName())