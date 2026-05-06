def _addFlushBatch(self):
        """
        Sends all waiting documents to Solr
        """

        if len(self._add_batch) > 0:
            language_batches = {}
            # Create command JSONs for each of language endpoints
            for lang in self.endpoints:
                # Append documents with languages without endpoint to default endpoint
                document_jsons = ["\"add\":" + json.dumps(data) for data in self._add_batch
                                  if data['doc'].get("language", self.default_endpoint) == lang or (lang == self.default_endpoint and not self.endpoints.has_key(data['doc'].get("language", None)))]
                command_json = "{" + ",".join(document_jsons) + "}"
                language_batches[lang] = command_json
            # Solr requires for documents to be sent in { "add" : { "doc" : {...} }, "add": { "doc" : { ... }, ... }
            # format which isn't possible with python dictionaries
            for lang in language_batches:
                self._send_solr_command(self.endpoints[lang], language_batches[lang])
                self._add_batch = []