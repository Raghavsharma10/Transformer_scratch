def delete(self, doc_id: str) -> bool:
        """Delete a document with id."""

        try:
            self.instance.delete(self.index, self.doc_type, doc_id)
        except RequestError as ex:
            logging.error(ex)
            return False
        else:
            return True