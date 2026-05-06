def rebuild(cls, session, tree_id=None):
        """ This function rebuid tree.

        Args:
            session (:mod:`sqlalchemy.orm.session.Session`): SQLAlchemy session

        Kwargs:
            tree_id (int or str): id of tree, default None

        Example:

        * :mod:`sqlalchemy_mptt.tests.TestTree.test_rebuild`
        """

        trees = session.query(cls).filter_by(parent_id=None)
        if tree_id:
            trees = trees.filter_by(tree_id=tree_id)
        for tree in trees:
            cls.rebuild_tree(session, tree.tree_id)