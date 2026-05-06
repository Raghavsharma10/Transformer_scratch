def leftsibling_in_level(self):
        """ Node to the left of the current node at the same level

        For example see
        :mod:`sqlalchemy_mptt.tests.cases.get_tree.test_leftsibling_in_level`
        """  # noqa
        table = _get_tree_table(self.__mapper__)
        session = Session.object_session(self)
        current_lvl_nodes = session.query(table) \
            .filter_by(level=self.level).filter_by(tree_id=self.tree_id) \
            .filter(table.c.lft < self.left).order_by(table.c.lft).all()
        if current_lvl_nodes:
            return current_lvl_nodes[-1]
        return None