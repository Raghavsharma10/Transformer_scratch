def _get_query_uri(self):
        """
        Returns the URI endpoint for performing queries of a
        Predix Time Series instance from environment inspection.
        """
        if 'VCAP_SERVICES' in os.environ:
            services = json.loads(os.getenv('VCAP_SERVICES'))
            predix_timeseries = services['predix-timeseries'][0]['credentials']
            return predix_timeseries['query']['uri'].partition('/v1')[0]
        else:
            return predix.config.get_env_value(self, 'query_uri')