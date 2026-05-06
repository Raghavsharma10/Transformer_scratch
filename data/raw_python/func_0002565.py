def mptt_before_insert(mapper, connection, instance):
    """ Based on example
    https://bitbucket.org/zzzeek/sqlalchemy/src/73095b353124/examples/nested_sets/nested_sets.py?at=master
    """
    table = _get_tree_table(mapper)
    db_pk = instance.get_pk_column()
    table_pk = getattr(table.c, db_pk.name)

    if instance.parent_id is None:
        instance.left = 1
        instance.right = 2
        instance.level = instance.get_default_level()
        tree_id = connection.scalar(
            select(
                [
                    func.max(table.c.tree_id) + 1
                ]
            )
        ) or 1
        instance.tree_id = tree_id
    else:
        (parent_pos_left,
         parent_pos_right,
         parent_tree_id,
         parent_level) = connection.execute(
            select(
                [
                    table.c.lft,
                    table.c.rgt,
                    table.c.tree_id,
                    table.c.level
                ]
            ).where(
                table_pk == instance.parent_id
            )
        ).fetchone()

        # Update key of right side
        connection.execute(
            table.update(
                and_(table.c.rgt >= parent_pos_right,
                     table.c.tree_id == parent_tree_id)
            ).values(
                lft=case(
                    [
                        (
                            table.c.lft > parent_pos_right,
                            table.c.lft + 2
                        )
                    ],
                    else_=table.c.lft
                ),
                rgt=case(
                    [
                        (
                            table.c.rgt >= parent_pos_right,
                            table.c.rgt + 2
                        )
                    ],
                    else_=table.c.rgt
                )
            )
        )

        instance.level = parent_level + 1
        instance.tree_id = parent_tree_id
        instance.left = parent_pos_right
        instance.right = parent_pos_right + 1