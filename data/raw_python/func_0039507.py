def get_documents(self, doc_format='dict'):
        """ Get the documents returned from Storege in this response.

            Keyword args:
                doc_format -- Specifies the doc_format for the returned documents.
                    Can be 'dict', 'etree' or 'string'. Default is 'dict'.

            Returns:
                A dict where keys are document ids and values depending of the required doc_format:
                    A dict representations of documents (see etree_to_dict());
                    A etree Element representing the document;
                    A raw XML document string.

            Raises:
                ParameterError -- The doc_format value is not allowed.
        """
        def get_doc_id(root, rel_path):
            if not rel_path:
                return root.text
            else:
                child = root.find(rel_path[0])
                if child is None:
                    return None
                return get_doc_id(child, rel_path[1:])

        if doc_format == 'dict':
            return dict([(get_doc_id(document, self._id_xpath), etree_to_dict(document)['document']) for
                        document in self._get_doc_list()])
        elif doc_format == 'etree':
            return dict([(get_doc_id(document, self._id_xpath), document) for
                        document in self._get_doc_list()])
        elif doc_format == 'list-etree':
            return self._get_doc_list()
        elif doc_format == 'list-string':
            return list([(ET.tostring(document)) for
                        document in self._get_doc_list()])
        elif doc_format in ('', None, 'string'):
            return dict([(get_doc_id(document, self._id_xpath), ET.tostring(document)) for
                        document in self._get_doc_list()])
        else:
            raise ParameterError("doc_format=" + doc_format)