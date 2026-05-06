def verify_document(self, document: Document) -> bool:
        """
        Check specified document
        :param duniterpy.documents.Document document:
        :return:
        """
        signature = base64.b64decode(document.signatures[0])
        prepended = signature + bytes(document.raw(), 'ascii')

        try:
            self.verify(prepended)
            return True
        except ValueError:
            return False