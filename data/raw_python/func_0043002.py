def search_obsgroups(self, search):
        """
        Search for :class:`meteorpi_model.ObservationGroup` entities

        :param search:
            an instance of :class:`meteorpi_model.ObservationGroupSearch` used to constrain the observations returned
            from the DB
        :return:
            a structure of {count:int total rows of an unrestricted search, observations:list of
            :class:`meteorpi_model.ObservationGroup`}
        """
        b = search_obsgroups_sql_builder(search)
        sql = b.get_select_sql(columns='g.uid, g.time, g.setAtTime, g.setByUser, g.publicId, g.title,'
                                       's.name AS semanticType',
                               skip=search.skip,
                               limit=search.limit,
                               order='g.time DESC')
        obs_groups = list(self.generators.obsgroup_generator(sql=sql, sql_args=b.sql_args))
        rows_returned = len(obs_groups)
        total_rows = rows_returned + search.skip
        if (rows_returned == search.limit > 0) or (rows_returned == 0 and search.skip > 0):
            self.con.execute(b.get_count_sql(), b.sql_args)
            total_rows = self.con.fetchone()['COUNT(*)']
        return {"count": total_rows,
                "obsgroups": obs_groups}