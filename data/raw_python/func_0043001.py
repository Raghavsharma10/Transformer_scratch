def get_obsgroup(self, group_id):
        """
        Retrieve an existing :class:`meteorpi_model.ObservationGroup` by its ID

        :param string group_id:
            UUID of the observation
        :return:
            A :class:`meteorpi_model.Observation` instance, or None if not found
        """
        search = mp.ObservationGroupSearch(group_id=group_id)
        b = search_obsgroups_sql_builder(search)
        sql = b.get_select_sql(columns='g.uid, g.time, g.setAtTime, g.setByUser, g.publicId, g.title,'
                                       's.name AS semanticType',
                               skip=0, limit=1, order='g.time DESC')
        obs_groups = list(self.generators.obsgroup_generator(sql=sql, sql_args=b.sql_args))
        if not obs_groups:
            return None
        return obs_groups[0]