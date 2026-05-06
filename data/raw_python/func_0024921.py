def get_datapoints(self, tags, start=None, end=None, order=None,
            limit=None, qualities=None, attributes=None, measurement=None,
            aggregations=None, post=False):
        """
        Returns all of the datapoints that match the given query.

            - tags: list or string identifying the name/tag (ie. "temp")
            - start: data after this, absolute or relative (ie. '1w-ago' or
              1494015972386)
            - end: data before this value
            - order: ascending (asc) or descending (desc)
            - limit: only return a few values (ie. 25)
            - qualities: data quality value (ie. [ts.GOOD, ts.UNCERTAIN])
            - attributes: dictionary of key-values (ie. {'unit': 'mph'})
            - measurement: tuple of operation and value (ie. ('gt', 30))
            - aggregations: summary statistics on data results (ie. 'avg')
            - post: POST query instead of GET (caching implication)

        A few additional observations:
            - allow service to do most data validation
            - order is applied before limit so resultset will differ

        The returned results match what the service response is so you'll
        need to unpack it as appropriate.  Oftentimes what you want for
        a simple single tag query will be:

            response['tags'][0]['results'][0]['values']

        """
        params = {}

        # Documentation says start is required for GET but not POST, but
        # seems to be required all the time, so using sensible default.
        if not start:
            start = '1w-ago'
            logging.warning("Defaulting query for data with start date %s" % (start))

        # Start date can be absolute or relative, only certain legal values
        # but service will throw error if used improperly.  (ms, s, mi, h, d,
        # w, mm, y).  Relative dates must end in -ago.
        params['start'] = start

        # Docs say when making POST with a start that end must also be
        # specified, but this does not seem to be the case.
        if end:
            # MAINT: error when end < start which is handled by service
            params['end'] = end

        params['tags'] = []
        if not isinstance(tags, list):
            tags = [tags]

        for tag in tags:
            query = {}
            query['name'] = tag

            # Limit resultset with an integer value
            if limit:
                query['limit'] = int(limit)

            # Order must be 'asc' or 'desc' but will get sensible error
            # from service.
            if order:
                query['order'] = order

            # Filters are complex and support filtering by
            # quality, measurement, and attributes.
            filters = {}

            # Check for the quality of the datapoints
            if qualities is not None:
                if isinstance(qualities, int) or isinstance(qualities, str):
                    qualities = [qualities]

                # Timeseries expects quality to be a string, not integer,
                # so coerce each into a string
                for i, quality in enumerate(qualities):
                    qualities[i] = str(quality)

                filters['qualities'] = {"values": qualities}

            # Check for attributes on the datapoints, expected to be
            # a dictionary of key / value pairs that datapoints must match.
            if attributes is not None:
                if not isinstance(attributes, dict):
                    raise ValueError("Attribute filters must be dictionary.")

                filters['attributes'] = attributes

            # Check for measurements that meets a given comparison operation
            # such as ge, gt, eq, ne, le, lt
            if measurement is not None:
                filters['measurements'] = {
                        'condition': measurement[0],
                        'values': measurement[1]
                        }

            # If we found any filters add them to the query
            if filters:
                query['filters'] = filters

            # Handle any additional aggregations of dataset
            if aggregations is not None:
                if not isinstance(aggregations, list):
                    aggregations = [aggregations]

                query['aggregations'] = []
                for aggregation in aggregations:
                    query['aggregations'].append({
                        'sampling': {'datapoints': 1},
                        'type': aggregation })

            params['tags'].append(query)

        if post:
            return self._post_datapoints(params)
        else:
            return self._get_datapoints({"query": json.dumps(params)})