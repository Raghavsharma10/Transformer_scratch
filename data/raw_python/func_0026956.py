def path_upper(
        cls, object_id, limit_depth=1000000, db_session=None, *args, **kwargs
    ):
        """
        This returns you path to root node starting from object_id
            currently only for postgresql

        :param object_id:
        :param limit_depth:
        :param db_session:
        :return:
        """
        tablename = cls.model.__table__.name
        raw_q = """
            WITH RECURSIVE subtree AS (
                    SELECT res.*, 1 as depth FROM {tablename} res
                    WHERE res.resource_id = :resource_id
                  UNION ALL
                    SELECT res_u.*, depth+1 as depth
                    FROM {tablename} res_u, subtree st
                    WHERE res_u.resource_id = st.parent_id
            )
            SELECT * FROM subtree WHERE depth<=:depth;
        """.format(
            tablename=tablename
        )
        db_session = get_db_session(db_session)
        q = (
            db_session.query(cls.model)
            .from_statement(sa.text(raw_q))
            .params(resource_id=object_id, depth=limit_depth)
        )
        return q