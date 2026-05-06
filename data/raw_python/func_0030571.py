def _index_document(self, identifier, force=False):
        """ Adds identifier document to the index. """
        writer = self.index.writer()
        all_names = set([x['name'] for x in self.index.searcher().documents()])
        if identifier['name'] not in all_names:
            writer.add_document(**identifier)
            writer.commit()