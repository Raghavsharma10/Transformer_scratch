def write_document(self, name, document):
        """
        This function will write a document to an XML file.
        """
        with open(name, 'wb') as out:
            out.write(etree.tostring(document,
                                     encoding='utf-8',
                                     pretty_print=True))