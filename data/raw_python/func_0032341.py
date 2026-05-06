def iter_funders(self):
        """Get a converted list of Funders as JSON dict."""
        root = self.doc_root
        funders = root.findall('./skos:Concept', namespaces=self.namespaces)
        for funder in funders:
            funder_json = self.fundrefxml2json(funder)
            yield funder_json