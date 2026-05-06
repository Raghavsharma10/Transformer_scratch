def fundertree2json(self, tree, oai_id):
        """Convert OpenAIRE's funder XML to JSON."""
        try:
            tree = self.get_subtree(tree, 'fundingtree')[0]
        except IndexError:  # pragma: nocover
            pass

        funder_node = self.get_subtree(tree, 'funder')
        subfunder_node = self.get_subtree(tree, '//funding_level_0')

        funder_id = self.get_text_node(funder_node[0], './id') \
            if funder_node else None
        subfunder_id = self.get_text_node(subfunder_node[0], './id') \
            if subfunder_node else None
        funder_name = self.get_text_node(funder_node[0], './shortname') \
            if funder_node else ""
        subfunder_name = self.get_text_node(subfunder_node[0], './name') \
            if subfunder_node else ""

        # Try to resolve the subfunder first, on failure try to resolve the
        # main funder, on failure raise an error.
        funder_doi_url = None
        if subfunder_id:
            funder_doi_url = self.funder_resolver.resolve_by_id(subfunder_id)
        if not funder_doi_url:
            if funder_id:
                funder_doi_url = self.funder_resolver.resolve_by_id(funder_id)
        if not funder_doi_url:
            funder_doi_url = self.funder_resolver.resolve_by_oai_id(oai_id)
        if not funder_doi_url:
            raise FunderNotFoundError(oai_id, funder_id, subfunder_id)

        funder_doi = FundRefDOIResolver.strip_doi_host(funder_doi_url)
        if not funder_name:
            # Grab name from FundRef record.
            resolver = Resolver(
                pid_type='frdoi', object_type='rec', getter=Record.get_record)
            try:
                dummy_pid, funder_rec = resolver.resolve(funder_doi)
                funder_name = funder_rec['acronyms'][0]
            except PersistentIdentifierError:
                raise OAIRELoadingError(
                    "Please ensure that funders have been loaded prior to"
                    "loading grants. Could not resolve funder {0}".format(
                        funder_doi))

        return dict(
            doi=funder_doi,
            url=funder_doi_url,
            name=funder_name,
            program=subfunder_name,
        )