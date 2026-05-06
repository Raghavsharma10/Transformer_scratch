def get_pmids(self, entry):
        """
        get `models.Pmid` objects from XML node entry

        :param entry: XML node entry
        :return: list of :class:`pyuniprot.manager.models.Pmid` objects
        """
        pmids = []

        for citation in entry.iterfind("./reference/citation"):

            for pubmed_ref in citation.iterfind('dbReference[@type="PubMed"]'):

                pmid_number = pubmed_ref.get('id')

                if pmid_number in self.pmids:

                    pmid_sqlalchemy_obj = self.session.query(models.Pmid)\
                        .filter(models.Pmid.pmid == pmid_number).one()

                    pmids.append(pmid_sqlalchemy_obj)

                else:
                    pmid_dict = citation.attrib
                    if not re.search('^\d+$', pmid_dict['volume']):
                        pmid_dict['volume'] = -1

                    del pmid_dict['type'] # not needed because already filtered for PubMed

                    pmid_dict.update(pmid=pmid_number)
                    title_tag = citation.find('./title')

                    if title_tag is not None:
                        pmid_dict.update(title=title_tag.text)

                    pmid_sqlalchemy_obj = models.Pmid(**pmid_dict)
                    self.session.add(pmid_sqlalchemy_obj)
                    self.session.flush()

                    pmids.append(pmid_sqlalchemy_obj)

                    self.pmids |= set([pmid_number, ]) # extend the cache of identifiers

        return pmids