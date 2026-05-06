def delete(self, ids):
        """
        Method to undeploy pool's by their ids

        :param ids: Identifiers of deployed pool's
        :return: Empty Dict
        """
        url = build_uri_with_ids('api/v3/pool/deploy/%s/', ids)

        return super(ApiPoolDeploy, self).delete(url)