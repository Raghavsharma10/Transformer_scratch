def from_parent_deeper(
        cls, parent_id=None, limit_depth=1000000, db_session=None, *args, **kwargs
    ):
        """
        This returns you subtree of ordered objects relative
        to the start parent_id (currently only implemented in postgresql)

        :param resource_id:
        :param limit_depth:
        :param db_session:
        :return:
        """

        if parent_id:
            limiting_clause = "res.parent_id = :parent_id"
        else:
            limiting_clause = "res.parent_id is null"
        tablename = cls.model.__table__.name
        raw_q = """
            WITH RECURSIVE subtree AS (
                    SELECT res.*, 1 AS depth, LPAD(res.ordering::CHARACTER VARYING, 7, '0') AS sorting,
                    res.resource_id::CHARACTER VARYING AS path
                    FROM {tablename} AS res WHERE {limiting_clause}
                  UNION ALL
                    SELECT res_u.*, depth+1 AS depth,
                    (st.sorting::CHARACTER VARYING || '/' || LPAD(res_u.ordering::CHARACTER VARYING, 7, '0') ) AS sorting,
                    (st.path::CHARACTER VARYING || '/' || res_u.resource_id::CHARACTER VARYING ) AS path
                    FROM {tablename} res_u, subtree st
                    WHERE res_u.parent_id = st.resource_id
            )
            SELECT * FROM subtree WHERE depth<=:depth ORDER BY sorting;
        """.format(
            tablename=tablename, limiting_clause=limiting_clause
        )  # noqa
        db_session = get_db_session(db_session)
        text_obj = sa.text(raw_q)
        query = db_session.query(cls.model, "depth", "sorting", "path")
        query = query.from_statement(text_obj)
        query = query.params(parent_id=parent_id, depth=limit_depth)
        return query