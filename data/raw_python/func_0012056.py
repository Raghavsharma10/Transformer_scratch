def get(self, identifier):
        """Fetch document by _id.

        Returns None if it is not found. (Will log a warning if not found as well. Should not be used
        to search an id.)"""
        logging.info('Download document with id ' + str(identifier) + '.')
        try:
            record = self.instance.get(index=self.index, doc_type=self.doc_type, id=identifier)
            if '_source' in record:
                return record['_source']
            else:
                return record
        except NotFoundError:
            return None