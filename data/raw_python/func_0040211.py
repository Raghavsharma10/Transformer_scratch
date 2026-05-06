def update_document_indicators(self, doc_id, citations, accesses):
        """
            Atualiza os indicadores de acessos e citações de um determinado
            doc_id.
            exemplo de doc_id: S0021-25712009000400007-spa
        """

        headers = {'content-type': 'application/json'}

        data = {
            "add": {
                "doc": {
                    "id": doc_id
                }
            }
        }

        if citations:
            data['add']['doc']['total_received'] = {'set': str(citations)}

        if accesses:
            data['add']['doc']['total_access'] = {'set': str(accesses)}

        params = {'wt': 'json'}

        response = self._do_request(
            self.UPDATE_ENDPOINT,
            params=params,
            data=json.dumps(data),
            headers=headers
        )

        if not response:
            logger.debug('Document (%s) could not be updated' % doc_id)

        logger.debug('Document (%s) updated' % doc_id)