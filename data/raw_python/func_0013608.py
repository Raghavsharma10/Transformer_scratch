def toProtocolElement(self):
        """
        Returns the GA4GH protocol representation of this ReferenceSet.
        """
        ret = protocol.ReferenceSet()
        ret.assembly_id = pb.string(self.getAssemblyId())
        ret.description = pb.string(self.getDescription())
        ret.id = self.getId()
        ret.is_derived = self.getIsDerived()
        ret.md5checksum = self.getMd5Checksum()
        if self.getSpecies():
            term = protocol.fromJson(
                json.dumps(self.getSpecies()), protocol.OntologyTerm)
            ret.species.term_id = term.term_id
            ret.species.term = term.term
        ret.source_accessions.extend(self.getSourceAccessions())
        ret.source_uri = pb.string(self.getSourceUri())
        ret.name = self.getLocalId()
        self.serializeAttributes(ret)
        return ret