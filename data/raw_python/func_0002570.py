def move_before(self, node_id):
        """ Moving one node of tree before another

        For example see:

        * :mod:`sqlalchemy_mptt.tests.cases.move_node.test_move_before_function`
        * :mod:`sqlalchemy_mptt.tests.cases.move_node.test_move_before_to_other_tree`
        * :mod:`sqlalchemy_mptt.tests.cases.move_node.test_move_before_to_top_level`
        """  # noqa
        session = Session.object_session(self)
        table = _get_tree_table(self.__mapper__)
        pk = getattr(table.c, self.get_pk_column().name)
        node = session.query(table).filter(pk == node_id).one()
        self.parent_id = node.parent_id
        self.mptt_move_before = node_id
        session.add(self)