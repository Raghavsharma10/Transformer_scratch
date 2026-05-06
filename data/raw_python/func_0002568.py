def move_inside(self, parent_id):
        """ Moving one node of tree inside another

        For example see:

        * :mod:`sqlalchemy_mptt.tests.cases.move_node.test_move_inside_function`
        * :mod:`sqlalchemy_mptt.tests.cases.move_node.test_move_inside_to_the_same_parent_function`
        """  # noqa
        session = Session.object_session(self)
        self.parent_id = parent_id
        self.mptt_move_inside = parent_id
        session.add(self)