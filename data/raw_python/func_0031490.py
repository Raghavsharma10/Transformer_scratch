def export_obo(self, path_to_export_file, name_of_ontology="uniprot", taxids=None):
        """
        export complete database to OBO (http://www.obofoundry.org/) file

        :param path_to_export_file: path to export file
        :param taxids: NCBI taxonomy identifiers to export (optional)
        """

        fd = open(path_to_export_file, 'w')

        header = "format-version: 0.1\ndata: {}\n".format(time.strftime("%d:%m:%Y %H:%M"))
        header += "ontology: {}\n".format(name_of_ontology)
        header += 'synonymtypedef: GENE_NAME "GENE NAME"\nsynonymtypedef: ALTERNATIVE_NAME "ALTERNATIVE NAME"\n'

        fd.write(header)

        query = self.session.query(models.Entry).limit(100)

        if taxids:
            query = query.filter(models.Entry.taxid.in_(taxids))

        for entry in query.all():

            fd.write('\n[Term]\nid: SWISSPROT:{}\n'.format(entry.accessions[0]))

            if len(entry.accessions) > 1:
                for accession in entry.accessions[1:]:
                    fd.write('alt_id: {}\n'.format(accession))

            fd.write('name: {}\n'.format(entry.recommended_full_name))

            for alternative_full_name in entry.alternative_full_names:
                fd.write('synonym: "{}" EXACT ALTERNATIVE_NAME []\n'.format(alternative_full_name.name))

            for alternative_short_name in entry.alternative_short_names:
                fd.write('synonym: "{}" EXACT ALTERNATIVE_NAME []\n'.format(alternative_short_name.name))

            fd.write('synonym: "{}" EXACT GENE_NAME []\n'.format(entry.gene_name))

            for xref in entry.db_references:
                if xref.type_ in ['GO', 'HGNC']:
                    xref.identifier = ':'.join(xref.identifier.split(':')[1:])
                fd.write('xref: {}:{}\n'.format(xref.type_, xref.identifier.replace('\\', '\\\\')))

        fd.close()