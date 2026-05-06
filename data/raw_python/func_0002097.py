def get_between_times(self, t1, t2, target=None):
        """
        Query for OPUS data between times t1 and t2.

        Parameters
        ----------
        t1, t2 : datetime.datetime, strings
            Start and end time for the query. If type is datetime, will be
            converted to isoformat string. If type is string already, it needs
            to be in an accepted international format for time strings.
        target : str
            Potential target for the observation query. Most likely will reduce
            the amount of data matching the query a lot.

        Returns
        -------
        None, but set's state of the object to have new query results stored
        in self.obsids.
        """
        try:
            # checking if times have isoformat() method (datetimes have)
            t1 = t1.isoformat()
            t2 = t2.isoformat()
        except AttributeError:
            # if not, should already be a string, so do nothing.
            pass
        myquery = self._get_time_query(t1, t2)
        if target is not None:
            myquery["target"] = target
        self.create_files_request(myquery, fmt="json")
        self.unpack_json_response()