def insertReference(self, reference):
        """
        Inserts the specified reference into this repository.
        """
        models.Reference.create(
            id=reference.getId(),
            referencesetid=reference.getParentContainer().getId(),
            name=reference.getLocalId(),
            length=reference.getLength(),
            isderived=reference.getIsDerived(),
            species=json.dumps(reference.getSpecies()),
            md5checksum=reference.getMd5Checksum(),
            sourceaccessions=json.dumps(reference.getSourceAccessions()),
            sourceuri=reference.getSourceUri())