def rebuild_tree(cls, session, tree_id):
        """ This method rebuid tree.

        Args:
            session (:mod:`sqlalchemy.orm.session.Session`): SQLAlchemy session
            tree_id (int or str): id of tree

        Example:

        * :mod:`sqlalchemy_mptt.tests.cases.get_tree.test_rebuild`
        """
        session.query(cls).filter_by(tree_id=tree_id)\
            .update({cls.left: 0, cls.right: 0, cls.level: 0})
        top = session.query(cls).filter_by(parent_id=None)\
            .filter_by(tree_id=tree_id).one()
        top.left = left = 1
        top.right = right = 2
        top.level = level = cls.get_default_level()

        def recursive(children, left, right, level):
            level = level + 1
            for i, node in enumerate(children):
                same_level_right = children[i - 1].right
                left = left + 1

                if i > 0:
                    left = left + 1
                if same_level_right:
                    left = same_level_right + 1

                right = left + 1
                node.left = left
                node.right = right
                parent = node.parent

                j = 0
                while parent:
                    parent.right = right + 1 + j
                    parent = parent.parent
                    j += 1

                node.level = level
                recursive(node.children, left, right, level)

        recursive(top.children, left, right, level)