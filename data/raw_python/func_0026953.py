def delete_branch(cls, resource_id=None, db_session=None, *args, **kwargs):
        """
        This deletes whole branch with children starting from resource_id

        :param resource_id:
        :param db_session:
        :return:
        """
        tablename = cls.model.__table__.name

        # lets lock rows to prevent bad tree states
        resource = ResourceService.lock_resource_for_update(
            resource_id=resource_id, db_session=db_session
        )
        parent_id = resource.parent_id
        ordering = resource.ordering
        raw_q = """
            WITH RECURSIVE subtree AS (
                    SELECT res.resource_id
                    FROM {tablename} AS res WHERE res.resource_id = :resource_id
                  UNION ALL
                    SELECT res_u.resource_id
                    FROM {tablename} res_u, subtree st
                    WHERE res_u.parent_id = st.resource_id
            )
            DELETE FROM resources where resource_id in (select * from subtree);
        """.format(
            tablename=tablename
        )  # noqa
        db_session = get_db_session(db_session)
        text_obj = sa.text(raw_q)
        db_session.execute(text_obj, params={"resource_id": resource_id})
        cls.shift_ordering_down(parent_id, ordering, db_session=db_session)
        return True