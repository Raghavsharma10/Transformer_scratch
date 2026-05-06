def query(self, start_date=None, end_date=None, ndays=None, metrics=[], dimensions=[], filters=None, segment=None, sort=[], start_index=1, max_results=10):
        """
        Execute a query.
        """
        if start_date:
            start_date = start_date
        elif getattr(self.args, 'start-date'):
            start_date = getattr(self.args, 'start-date')
        elif self.config.get('start-date', None):
            start_date = self.config['start-date']
        else:
            start_date = '2005-01-01'

        if end_date:
            end_date = end_date
        elif getattr(self.args, 'end-date'):
            end_date = getattr(self.args, 'end-date')
        elif self.config.get('end-date', None):
            end_date = self.config['end-date']
        elif ndays:
            end_date = self._ndays(start_date, ndays)
        elif self.args.ndays:
            end_date = self._ndays(start_date, self.args.ndays)
        elif self.config.get('ndays', None):
            end_date = self._ndays(start_date, self.config['ndays'])
        else:
            end_date = 'today' 

        if self.args.domain:
            domain = self.args.domain
        elif self.config.get('domain', None):
            domain = self.config['domain']
        else:
            domain = None

        if domain:
            domain_filter = 'ga:hostname==%s' % domain

            if filters:
                filters = '%s;%s' % (domain_filter, filters)
            else:
                filters = domain_filter

        if self.args.prefix:
            prefix = self.args.prefix
        elif self.config.get('prefix', None):
            prefix = self.config['prefix']
        else:
            prefix = None

        if prefix:
            prefix_filter = 'ga:pagePath=~^%s' % prefix 
                
            if filters:
                filters = '%s;%s' % (prefix_filter, filters)
            else:
                filters = prefix_filter

        return self.service.data().ga().get(
            ids='ga:' + self.config['property-id'],
            start_date=start_date,
            end_date=end_date,
            metrics=','.join(metrics) or None,
            dimensions=','.join(dimensions) or None,
            filters=filters,
            segment=segment,
            sort=','.join(sort) or None,
            start_index=str(start_index),
            max_results=str(max_results)
        ).execute()