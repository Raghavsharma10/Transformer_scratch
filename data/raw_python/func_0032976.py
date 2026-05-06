def complete(self):
        """ Check, if out hashed date:url id is already in the index. """
        id = hashlib.sha1('%s:%s' % (self.date, self.url)).hexdigest()
        es = elasticsearch.Elasticsearch()
        try:
            es.get(index='frontpage', doc_type='html', id=id)
        except elasticsearch.NotFoundError:
            return False
        return True