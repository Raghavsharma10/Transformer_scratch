def create(self, ids):
        """
        Method to deploy pool's

        :param pools: Identifiers of pool's desired to be deployed
        :return: Empty Dict
        """

        url = build_uri_with_ids('api/v3/pool/deploy/%s/', ids)
        return super(ApiPoolDeploy, self).post(url)