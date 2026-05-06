def _call(self, target, method, target_class=None, single_result=True, raw=False, files=None, **kwargs):
        """
        Low-level call to HasOffers API.
        :param target_class: type of resulting object/objects.
        """
        if target_class is None:
            target_class = target
        params = prepare_query_params(
            NetworkToken=self.network_token,
            NetworkId=self.network_id,
            Target=target,
            Method=method,
            **kwargs
        )
        kwargs = {'url': self.endpoint, 'params': params, 'verify': self.verify, 'method': 'GET'}
        if files:
            kwargs.update({'method': 'POST', 'files': files})

        self.logger.debug('Request parameters: %s', params)
        response = self.session.request(**kwargs)

        self.logger.debug('Response [%s]: %s', response.status_code, response.text)
        response.raise_for_status()
        data = response.json(object_pairs_hook=OrderedDict)
        return self.handle_response(data, target=target_class, single_result=single_result, raw=raw)