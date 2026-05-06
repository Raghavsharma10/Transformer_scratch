def _get_query_zone_id(self):
        """
        Returns the ZoneId for performing queries of a Predix
        Time Series instance from environment inspection.
        """
        if 'VCAP_SERVICES' in os.environ:
            services = json.loads(os.getenv('VCAP_SERVICES'))
            predix_timeseries = services['predix-timeseries'][0]['credentials']
            return predix_timeseries['query']['zone-http-header-value']
        else:
            return predix.config.get_env_value(self, 'query_zone_id')