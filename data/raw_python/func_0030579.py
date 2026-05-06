def _index_document(self, document, force=False):
        """ Adds parition document to the index. """
        if force:
            self._delete(vid=document['vid'])

        writer = self.index.writer()
        writer.add_document(**document)
        writer.commit()