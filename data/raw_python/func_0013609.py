def toProtocolElement(self):
        """
        Returns the GA4GH protocol representation of this Reference.
        """
        reference = protocol.Reference()
        reference.id = self.getId()
        reference.is_derived = self.getIsDerived()
        reference.length = self.getLength()
        reference.md5checksum = self.getMd5Checksum()
        reference.name = self.getName()
        if self.getSpecies():
            term = protocol.fromJson(
                json.dumps(self.getSpecies()), protocol.OntologyTerm)
            reference.species.term_id = term.term_id
            reference.species.term = term.term
        reference.source_accessions.extend(self.getSourceAccessions())
        reference.source_divergence = pb.int(self.getSourceDivergence())
        reference.source_uri = self.getSourceUri()
        self.serializeAttributes(reference)
        return reference