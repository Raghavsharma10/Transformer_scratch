def read_obo(cls, path, flatten=True, part_of_cc_only=False):
        """ Parse an OBO file and store GO term information.

        Parameters
        ----------
        path: str
            Path of the OBO file.
        flatten: bool, optional
            If set to False, do not generate a list of all ancestors and
            descendants for each GO term.
        part_of_cc_only: bool, optional
            Legacy parameter for backwards compatibility. If set to True,
            ignore ``part_of`` relations outside the ``cellular_component``
            domain.

        Notes
        -----
        The OBO file must end with a line break.
        """

        name2id = {}
        alt_id = {}
        syn2id = {}
        terms = []

        with open(path) as fh:
            n = 0
            while True:
                try:
                    nextline = next(fh)
                except StopIteration:
                    break
                if nextline == '[Term]\n':
                    n += 1
                    id_ = next(fh)[4:-1]
                    # acc = get_acc(id_)
                    name = next(fh)[6:-1]
                    name2id[name] = id_
                    domain = next(fh)[11:-1]
                    def_ = None
                    is_a = set()
                    part_of = set()
                    l = next(fh)
                    while l != '\n':
                        if l.startswith('alt_id:'):
                            alt_id[l[8:-1]] = id_
                        elif l.startswith('def: '):
                            idx = l[6:].index('"')
                            def_ = l[6:(idx+6)]
                        elif l.startswith('is_a:'):
                            is_a.add(l[6:16])
                        elif l.startswith('synonym:'):
                            idx = l[10:].index('"')
                            if l[(10+idx+2):].startswith("EXACT"):
                                s = l[10:(10+idx)]
                                syn2id[s] = id_
                        elif l.startswith('relationship: part_of'):
                            if part_of_cc_only:
                                if domain == 'cellular_component':
                                    part_of.add(l[22:32])
                            else:
                                part_of.add(l[22:32])
                        l = next(fh)
                    assert def_ is not None
                    terms.append(GOTerm(id_, name, domain, def_, is_a, part_of))

        logger.info('Parsed %d GO term definitions.', n)

        ontology = cls(terms, syn2id, alt_id, name2id)

        # store children and parts
        logger.info('Adding child and part relationships...')
        for term in ontology:
            for parent in term.is_a:
                ontology[parent].children.add(term.id)
            for whole in term.part_of:
                ontology[whole].parts.add(term.id)

        if flatten:
            logger.info('Flattening ancestors...')
            ontology._flatten_ancestors()
            logger.info('Flattening descendants...')
            ontology._flatten_descendants()
            ontology._flattened = True

        return ontology