def index_into(self, document, id) -> bool:
        """Index a single document into the index."""
        try:
            self.instance.index(index=self.index, doc_type=self.doc_type, body=json.dumps(document, ensure_ascii=False), id=id)
        except RequestError as ex:
            logging.error(ex)
            return False
        else:
            return True