def from_resource_deeper(
        cls, resource_id=None, limit_depth=1000000, db_session=None, *args, **kwargs
    ):
        """
        This returns you subtree of ordered objects relative
        to the start resource_id (currently only implemented in postgresql)

        :param resource_id:
        :param limit_depth:
        :param db_session:
        :return:
        """
        tablename = cls.model.__table__.name
        raw_q = """
            WITH RECURSIVE subtree AS (
                    SELECT res.*, 1 AS depth, LPAD(res.ordering::CHARACTER VARYING, 7, '0') AS sorting,
                    res.resource_id::CHARACTER VARYING AS path
                    FROM {tablename} AS res WHERE res.resource_id = :resource_id
                  UNION ALL
                    SELECT res_u.*, depth+1 AS depth,
                    (st.sorting::CHARACTER VARYING || '/' || LPAD(res_u.ordering::CHARACTER VARYING, 7, '0') ) AS sorting,
                    (st.path::CHARACTER VARYING || '/' || res_u.resource_id::CHARACTER VARYING ) AS path
                    FROM {tablename} res_u, subtree st
                    WHERE res_u.parent_id = st.resource_id
            )
            SELECT * FROM subtree WHERE depth<=:depth ORDER BY sorting;
        """.format(
            tablename=tablename
        )  # noqa
        db_session = get_db_session(db_session)
        text_obj = sa.text(raw_q)
        query = db_session.query(cls.model, "depth", "sorting", "path")
        query = query.from_statement(text_obj)
        query = query.params(resource_id=resource_id, depth=limit_depth)
        return query