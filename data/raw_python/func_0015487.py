def rdf(self, rdf: Optional[Union[str, Graph]]) -> None:
        """ Set the RDF DataSet to be evaulated.  If ``rdf`` is a string, the presence of a return is the
        indicator that it is text instead of a location.

        :param rdf: File name, URL, representation of rdflib Graph
        """
        if isinstance(rdf, Graph):
            self.g = rdf
        else:
            self.g = Graph()
            if isinstance(rdf, str):
                if '\n' in rdf or '\r' in rdf:
                    self.g.parse(data=rdf, format=self.rdf_format)
                elif ':' in rdf:
                    self.g.parse(location=rdf, format=self.rdf_format)
                else:
                    self.g.parse(source=rdf, format=self.rdf_format)