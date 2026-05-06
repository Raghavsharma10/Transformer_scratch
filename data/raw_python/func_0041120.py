def set_documents(self, documents, fully_formed=False):
        """ Wrap documents in the correct root tags, add id fields and convert them to xml strings.

            Args:
                documents -- If fully_formed is False (default), accepts dict where keys are document ids and values can be ether
                            xml string, etree.ElementTree or dict representation of an xml document (see dict_to_etree()).
                            If fully_formed is True, accepts list or single document where ids are integrated in document or
                            not needed and document has the right root tag.

            Keyword args:
                fully_formed  -- If documents are fully formed (contains the right root tags and id fields) set to True
                            to avoid the owerhead of documets beeing parsed at all. If set to True only list of documents or
                            a single document can be pased as 'documents', not a dict of documents. Default is False.
        """
        def add_id(document, id):
            def make_id_tag(root, rel_path, max_depth):
                if max_depth < 0:
                    raise ParameterError("document_id_xpath too deep!")
                if not rel_path:
                    return root
                else:
                    child = root.find(rel_path[0])
                    if child is None:
                        child = ET.Element(rel_path[0])
                        root.append(child)
                    return make_id_tag(child, rel_path[1:], max_depth - 1)
            make_id_tag(document, doc_id_xpath, 10).text = str(id)

        if fully_formed: # documents is a list or single document that contians root tags and id fields.
            if not isinstance(documents, list):
                documents = [documents]
        else: # documents is dict with ids as keys and documents as values.
            doc_root_tag = self.connection.document_root_xpath  # Local scope is faster.
            doc_id_xpath = self.connection.document_id_xpath.split('/')
            # Convert to etrees.
            documents = dict([(id, to_etree((document if document is not None else
                                             query.term('', doc_root_tag)), doc_root_tag))
                             for id, document in documents.items()])     # TODO: possibly ineficient
            # If root not the same as given xpath, make new root and append to it.
            for id, document in documents.items():
                if document.tag != doc_root_tag:
                    documents[id] = ET.Element(doc_root_tag)
                    documents[id].append(document)  # documents is still the old reference
            # Insert ids in documents and collapse to a list of documents.
            for id, document in documents.items():
                add_id(document, id)
            documents = documents.values()
        self._documents = map(to_raw_xml, documents)