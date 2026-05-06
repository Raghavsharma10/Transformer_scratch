def move_after(self, node_id):
        """ Moving one node of tree after another

        For example see :mod:`sqlalchemy_mptt.tests.cases.move_node.test_move_after_function`
        """  # noqa
        session = Session.object_session(self)
        self.parent_id = self.parent_id
        self.mptt_move_after = node_id
        session.add(self)