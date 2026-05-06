def index_documents(self, fresh_docs, model):
        """
        Update fresh index with new documents (potentially replacing old ones with
        the same id). `fresh_docs` is a dictionary-like object (=dict, sqlitedict, shelve etc)
        that maps document_id->document.
        """
        docids = fresh_docs.keys()
        vectors = (model.docs2vecs(fresh_docs[docid] for docid in docids))
        logger.info("adding %i documents to %s" % (len(docids), self))
        self.qindex.add_documents(vectors)
        self.qindex.save()
        self.update_ids(docids)