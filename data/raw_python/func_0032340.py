def fundrefxml2json(self, node):
        """Convert a FundRef 'skos:Concept' node into JSON."""
        doi = FundRefDOIResolver.strip_doi_host(self.get_attrib(node,
                                                'rdf:about'))
        oaf_id = FundRefDOIResolver().resolve_by_doi(
            "http://dx.doi.org/" + doi)
        name = node.find('./skosxl:prefLabel/skosxl:Label/skosxl:literalForm',
                         namespaces=self.namespaces).text
        # Extract acronyms
        acronyms = []
        for n in node.findall('./skosxl:altLabel/skosxl:Label',
                              namespaces=self.namespaces):
            usagenode = n.find('./fref:usageFlag', namespaces=self.namespaces)
            if usagenode is not None:
                if self.get_attrib(usagenode, 'rdf:resource') == \
                        ('http://data.crossref.org/fundingdata'
                         '/vocabulary/abbrevName'):
                    label = n.find('./skosxl:literalForm',
                                   namespaces=self.namespaces)
                    if label is not None:
                        acronyms.append(label.text)

        parent_node = node.find('./skos:broader', namespaces=self.namespaces)
        if parent_node is None:
            parent = {}
        else:
            parent = {
                "$ref": self.get_attrib(parent_node, 'rdf:resource'),
            }
        country_elem = node.find('./svf:country', namespaces=self.namespaces)
        country_url = self.get_attrib(country_elem, 'rdf:resource')
        country_code = self.cc_resolver.cc_from_url(country_url)
        type_ = node.find('./svf:fundingBodyType',
                          namespaces=self.namespaces).text
        subtype = node.find('./svf:fundingBodySubType',
                            namespaces=self.namespaces).text
        country_elem = node.find('./svf:country', namespaces=self.namespaces)

        modified_elem = node.find('./dct:modified', namespaces=self.namespaces)
        created_elem = node.find('./dct:created', namespaces=self.namespaces)

        json_dict = {
            '$schema': self.schema_formatter.schema_url,
            'doi': doi,
            'identifiers': {
                'oaf': oaf_id,
            },
            'name': name,
            'acronyms': acronyms,
            'parent': parent,
            'country': country_code,
            'type': type_,
            'subtype': subtype.lower(),
            'remote_created': (created_elem.text if created_elem is not None
                               else None),
            'remote_modified': (modified_elem.text if modified_elem is not None
                                else None),
        }
        return json_dict