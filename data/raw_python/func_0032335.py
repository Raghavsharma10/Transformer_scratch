def grantxml2json(self, grant_xml):
        """Convert OpenAIRE grant XML into JSON."""
        tree = etree.fromstring(grant_xml)
        # XML harvested from OAI-PMH has a different format/structure
        if tree.prefix == 'oai':
            ptree = self.get_subtree(
                tree, '/oai:record/oai:metadata/oaf:entity/oaf:project')[0]
            header = self.get_subtree(tree, '/oai:record/oai:header')[0]
            oai_id = self.get_text_node(header, 'oai:identifier')
            modified = self.get_text_node(header, 'oai:datestamp')
        else:
            ptree = self.get_subtree(
                tree, '/record/result/metadata/oaf:entity/oaf:project')[0]
            header = self.get_subtree(tree, '/record/result/header')[0]
            oai_id = self.get_text_node(header, 'dri:objIdentifier')
            modified = self.get_text_node(header, 'dri:dateOfTransformation')

        url = self.get_text_node(ptree, 'websiteurl')
        code = self.get_text_node(ptree, 'code')
        title = self.get_text_node(ptree, 'title')
        acronym = self.get_text_node(ptree, 'acronym')
        startdate = self.get_text_node(ptree, 'startdate')
        enddate = self.get_text_node(ptree, 'enddate')

        funder = self.fundertree2json(ptree, oai_id)

        internal_id = "{0}::{1}".format(funder['doi'], code)
        eurepo_id = \
            "info:eu-repo/grantAgreement/{funder}/{program}/{code}/".format(
                funder=quote_plus(funder['name'].encode('utf8')),
                program=quote_plus(funder['program'].encode('utf8')),
                code=quote_plus(code.encode('utf8')), )

        ret_json = {
            '$schema': self.schema_formatter.schema_url,
            'internal_id': internal_id,
            'identifiers': {
                'oaf': oai_id,
                'eurepo': eurepo_id,
                'purl': url if url.startswith("http://purl.org/") else None,
            },
            'code': code,
            'title': title,
            'acronym': acronym,
            'startdate': startdate,
            'enddate': enddate,
            'funder': {'$ref': funder['url']},
            'program': funder['program'],
            'url': url,
            'remote_modified': modified,
        }
        return ret_json