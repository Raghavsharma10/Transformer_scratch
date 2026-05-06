def set_metric(self, slug, value, category=None, expire=None, date=None):
        """Assigns a specific value to the *current* metric. You can use this
        to start a metric at a value greater than 0 or to reset a metric.

        The given slug will be used to generate Redis keys at the following
        granularities: Seconds, Minutes, Hours, Day, Week, Month, and Year.

        Parameters:

        * ``slug`` -- a unique value to identify the metric; used in
          construction of redis keys (see below).
        * ``value`` -- The value of the metric.
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
        keys = self._build_keys(slug, date=date)

        # Add the slug to the set of metric slugs
        self.r.sadd(self._metric_slugs_key, slug)

        # Construct a dictionary of key/values for use with mset
        data = {}
        for k in keys:
            data[k] = value
        self.r.mset(data)

        # Add the category if applicable.
        if category:
            self._categorize(slug, category)

        # Expire the Metric in ``expire`` seconds if applicable.
        if expire:
            for k in keys:
                self.r.expire(k, expire)