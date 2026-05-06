def environmentvip_step(self, finality='', client='', environmentp44=''):
        """
        List finality, client or environment vip list.
        Param finality: finality of environment(optional)
        Param client: client of environment(optional)
        Param environmentp44: environmentp44(optional)
        Return finality list: when request has no finality and client.
        Return client list: when request has only finality.
        Return list environment vip: when request has finality and client.
        Return environment vip: when request has finality, client and environmentvip
        """

        uri = 'api/v3/environment-vip/step/?finality=%s&client=%s&environmentp44=%s' % (
            finality, client, environmentp44)

        return super(ApiEnvironmentVip, self).get(
            uri)