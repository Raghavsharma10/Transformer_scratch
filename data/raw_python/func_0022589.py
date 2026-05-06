def metric(self, slug, num=1, category=None, expire=None, date=None):
        """Records a metric, creating it if it doesn't exist or incrementing it
        if it does. All metrics are prefixed with 'm', and automatically
        aggregate for Seconds, Minutes, Hours, Day, Week, Month, and Year.

        Parameters:

        * ``slug`` -- a unique value to identify the metric; used in
          construction of redis keys (see below).
        * ``num`` -- Set or Increment the metric by this number; default is 1.
        * ``category`` -- (optional) Assign the metric to a Category (a string)
        * ``expire`` -- (optional) Specify the number of seconds in which the
          metric will expire.
        * ``date`` -- (optional) Specify the timestamp for the metric; default
          used to build the keys will be the current date and time in UTC form.

        Redis keys for each metric (slug) take the form:

            m:<slug>:s:<yyyy-mm-dd-hh-mm-ss> # Second
            m:<slug>:i:<yyyy-mm-dd-hh-mm>    # Minute
            m:<slug>:h:<yyyy-mm-dd-hh>       # Hour
            m:<slug>:<yyyy-mm-dd>            # Day
            m:<slug>:w:<yyyy-num>            # Week (year - week number)
            m:<slug>:m:<yyyy-mm>             # Month
            m:<slug>:y:<yyyy>                # Year

        """
        # Add the slug to the set of metric slugs
        self.r.sadd(self._metric_slugs_key, slug)

        if category:
            self._categorize(slug, category)

        # Increment keys. NOTE: current redis-py (2.7.2) doesn't include an
        # incrby method; .incr accepts a second ``amount`` parameter.
        keys = self._build_keys(slug, date=date)

        # Use a pipeline to speed up incrementing multiple keys
        pipe = self.r.pipeline()
        for key in keys:
            pipe.incr(key, num)
            if expire:
                pipe.expire(key, expire)
        pipe.execute()