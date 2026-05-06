def response_complete(self):
        """
        Signal that this particular entity has been fully processed. The exporter will not send it to this target again
        under this particular export configuration (there is no guarantee another export configuration on the same
        server won't send it, or that it won't be received from another server though, so you must always check whether
        you have an entity and return this status as early as possible if so)

        :return:
            A response that can be returned from a Flask service method
        """
        ImportRequest.logger.info("Completed import for {0} with id {1}".format(self.entity_type, self.entity_id))
        ImportRequest.logger.debug("Sending: complete")
        return jsonify({'state': 'complete'})