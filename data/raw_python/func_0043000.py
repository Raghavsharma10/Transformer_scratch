def has_obsgroup_id(self, group_id):
        """
        Check for the presence of the given group_id

        :param string group_id:
            The group ID
        :return:
            True if we have a :class:`meteorpi_model.ObservationGroup` with this Id, False otherwise
        """
        self.con.execute('SELECT 1 FROM archive_obs_groups WHERE publicId = %s', (group_id,))
        return len(self.con.fetchall()) > 0