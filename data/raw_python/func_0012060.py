def script_update(self, script: str, params: Union[dict, None], doc_id: str):
        """Uses painless script to update a document.

        See documentation for more information.
        """
        body = {
            'script': {
                'source': script,
                'lang': 'painless'
            }
        }
        if params is not None:
            body['script']['params'] = params
        self.instance.update(self.index, self.doc_type, doc_id, body=body)