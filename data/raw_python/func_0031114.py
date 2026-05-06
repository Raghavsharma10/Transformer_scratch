def read_msigdb_xml(cls, path, entrez2gene, species=None):  # pragma: no cover
        """Read the complete MSigDB database from an XML file.

        The XML file can be downloaded from here:
        http://software.broadinstitute.org/gsea/msigdb/download_file.jsp?filePath=/resources/msigdb/5.0/msigdb_v5.0.xml

        Parameters
        ----------
        path: str
            The path name of the XML file.
        entrez2gene: dict or OrderedDict (str: str)
            A dictionary mapping Entrez Gene IDs to gene symbols (names).
        species: str, optional
            A species name (e.g., "Homo_sapiens"). Only gene sets for that
            species will be retained. (None)

        Returns
        -------
        GeneSetCollection
            The gene set database containing the MSigDB gene sets.
        """

        # note: is XML file really encoded in UTF-8?

        assert isinstance(path, (str, _oldstr))
        assert isinstance(entrez2gene, (dict, OrderedDict))
        assert species is None or isinstance(species, (str, _oldstr))

        logger.debug('Path: %s', path)
        logger.debug('entrez2gene type: %s', str(type(entrez2gene)))

        i = [0]
        gene_sets = []

        total_gs = [0]
        total_genes = [0]

        species_excl = [0]
        unknown_entrezid = [0]

        src = 'MSigDB'

        def handle_item(pth, item):
            # callback function for xmltodict.parse()

            total_gs[0] += 1
            data = pth[1][1]

            spec = data['ORGANISM']
            # filter by species
            if species is not None and spec != species:
                species_excl[0] += 1
                return True

            id_ = data['SYSTEMATIC_NAME']
            name = data['STANDARD_NAME']
            coll = data['CATEGORY_CODE']
            desc = data['DESCRIPTION_BRIEF']
            entrez = data['MEMBERS_EZID'].split(',')

            genes = []
            for e in entrez:
                total_genes[0] += 1
                try:
                    genes.append(entrez2gene[e])
                except KeyError:
                    unknown_entrezid[0] += 1

            if not genes:
                logger.warning('Gene set "%s" (%s) has no known genes!',
                               name, id_)
                return True

            gs = GeneSet(id_, name, genes, source=src,
                         collection=coll, description=desc)
            gene_sets.append(gs)
            i[0] += 1
            return True

        # parse the XML file using the xmltodict package
        with io.open(path, 'rb') as fh:
            xmltodict.parse(fh.read(), encoding='UTF-8', item_depth=2,
                            item_callback=handle_item)

        # report some statistics
        if species_excl[0] > 0:
            kept = total_gs[0] - species_excl[0]
            perc = 100 * (kept / float(total_gs[0]))
            logger.info('%d of all %d gene sets (%.1f %%) belonged to the '
                        'specified species.', kept, total_gs[0], perc)

        if unknown_entrezid[0] > 0:
            unkn = unknown_entrezid[0]
            # known = total_genes[0] - unknown_entrezid[0]
            perc = 100 * (unkn / float(total_genes[0]))
            logger.warning('%d of a total of %d genes (%.1f %%) had an ' +
                           'unknown Entrez ID.', unkn, total_genes[0], perc)

        logger.info('Parsed %d entries, resulting in %d gene sets.',
                    total_gs[0], len(gene_sets))

        return cls(gene_sets)