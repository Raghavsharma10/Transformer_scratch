def register_obsgroup(self, title, user_id, semantic_type, obs_time, set_time, obs=None, grp_meta=None):
        """
        Register a new observation, updating the database and returning the corresponding Observation object

        :param string title:
            The title of this observation group
        :param string user_id:
            The ID of the user who created this observation
        :param float obs_time:
            The UTC date/time of the observation
        :param float set_time:
            The UTC date/time that this group was created
        :param list obs:
            A list of :class: publicIds of observations which are members of this group
        :param list grp_meta:
            A list of :class:`meteorpi_model.Meta` used to provide additional information about this observation
        :return:
            The :class:`meteorpi_model.ObservationGroup` as stored in the database
        """

        if grp_meta is None:
            grp_meta = []

        # Create a unique ID for this observation
        group_id = mp.get_hash(set_time, title, user_id)

        # Get ID code for semantic_type
        semantic_type_id = self.get_obs_type_id(semantic_type)

        # Insert into database
        self.con.execute("""
INSERT INTO archive_obs_groups (publicId, title, time, setByUser, setAtTime, semanticType)
VALUES
(%s, %s, %s, %s, %s, %s);
""", (group_id, title, obs_time, user_id, set_time, semantic_type_id))

        # Store list of observations into the database
        for item in obs:
            self.con.execute("""
INSERT INTO archive_obs_group_members (groupId, observationId)
VALUES
((SELECT uid FROM archive_obs_groups WHERE publicId=%s), (SELECT uid FROM archive_observations WHERE publicId=%s));
""", (group_id, item))
        # Store the observation metadata
        for meta in grp_meta:
            self.set_obsgroup_metadata(user_id, group_id, meta, obs_time)

        obs_group = mp.ObservationGroup(group_id=group_id,
                                        title=title,
                                        obs_time=obs_time,
                                        user_id=user_id,
                                        set_time=set_time,
                                        semantic_type=semantic_type,
                                        obs_records=[],
                                        meta=grp_meta)
        return obs_group