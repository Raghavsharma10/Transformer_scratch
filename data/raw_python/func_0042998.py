def search_observations(self, search):
        """
        Search for :class:`meteorpi_model.Observation` entities

        :param search:
            an instance of :class:`meteorpi_model.ObservationSearch` used to constrain the observations returned from
            the DB
        :return:
            a structure of {count:int total rows of an unrestricted search, observations:list of
            :class:`meteorpi_model.Observation`}
        """
        b = search_observations_sql_builder(search)
        sql = b.get_select_sql(columns='l.publicId AS obstory_id, l.name AS obstory_name, '
                                       'o.obsTime, s.name AS obsType, o.publicId, o.uid',
                               skip=search.skip,
                               limit=search.limit,
                               order='o.obsTime DESC')
        obs = list(self.generators.observation_generator(sql=sql, sql_args=b.sql_args))
        rows_returned = len(obs)
        total_rows = rows_returned + search.skip
        if (rows_returned == search.limit > 0) or (rows_returned == 0 and search.skip > 0):
            self.con.execute(b.get_count_sql(), b.sql_args)
            total_rows = self.con.fetchone()['COUNT(*)']
        return {"count": total_rows,
                "obs": obs}