def response_continue(self):
        """
        Signals that a partial reception of data has occurred and that the exporter should continue to send data for
        this entity. This should also be used if import-side caching has missed, in which case the response will direct
        the exporter to re-send the full data for the entity (otherwise it will send back the entity ID and rely on the
        import party's caching to resolve it). Use this for generic cases where we need to be messaged again about this
        entity - currently used after requesting and receiving a status block, and in its cache-refresh form if we have
        a cache miss during import.

        :return:
            A response that can be returned from a Flask service method
        """
        if self.entity is not None:
            ImportRequest.logger.debug("Sending: continue")
            return jsonify({'state': 'continue'})
        else:
            ImportRequest.logger.debug("Sending: continue-nocache")
            return jsonify({'state': 'continue-nocache'})