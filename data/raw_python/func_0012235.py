def route(self, arg, destination=None, waypoints=None, raw=False, **kwargs):
        """
        Query a route.

        route(locations): points can be
            - a sequence of locations
            - a Shapely LineString
        route(origin, destination, waypoints=None)
             - origin and destination are a single destination
             - waypoints are the points to be inserted between the
               origin and destination
             
             If waypoints is specified, destination must also be specified

        Each location can be:
            - string (will be geocoded by the routing provider. Not all
              providers accept this as input)
            - (longitude, latitude) sequence (tuple, list, numpy array, etc.)
            - Shapely Point with x as longitude, y as latitude

        Additional parameters
        ---------------------
        raw : bool, default False
            Return the raw json dict response from the service

        Returns
        -------
        list of Route objects
        If raw is True, returns the json dict instead of converting to Route
        objects

        Examples
        --------
        mq = directions.Mapquest(key)
        routes = mq.route('1 magazine st. cambridge, ma', 
                          'south station boston, ma')

        routes = mq.route('1 magazine st. cambridge, ma', 
                          'south station boston, ma', 
                          waypoints=['700 commonwealth ave. boston, ma'])

        # Uses each point in the line as a waypoint. There is a limit to the
        # number of waypoints for each service. Consult the docs.
        line = LineString(...)
        routes = mq.route(line)  

        # Feel free to mix different location types
        routes = mq.route(line.coords[0], 'south station boston, ma',
                          waypoints=[(-71.103972, 42.349324)])

        """
        points = _parse_points(arg, destination, waypoints)
        if len(points) < 2:
            raise ValueError('You must specify at least 2 points')

        self.rate_limit_wait()
        data = self.raw_query(points, **kwargs)
        self._last_query = time.time()

        if raw:
            return data
        return self.format_output(data)