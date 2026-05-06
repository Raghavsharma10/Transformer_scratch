def insert_entries(self, entries_xml, taxids=None):
        """Inserts UniProt entries from XML

        :param str entries_xml: XML string
        :param Optional[list[int]] taxids: NCBI taxonomy IDs
        """

        entries = etree.fromstring(entries_xml)
        del entries_xml

        for entry in entries:
            self.insert_entry(entry, taxids)
            entry.clear()
            del entry

        entries.clear()
        del entries

        self.session.commit()