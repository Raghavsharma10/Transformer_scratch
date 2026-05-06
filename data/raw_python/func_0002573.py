def get_tree(cls, session=None, json=False, json_fields=None, query=None):
        """ This method generate tree of current node table in dict or json
        format. You can make custom query with attribute ``query``. By default
        it return all nodes in table.

        Args:
            session (:mod:`sqlalchemy.orm.session.Session`): SQLAlchemy session

        Kwargs:
            json (bool): if True return JSON jqTree format
            json_fields (function): append custom fields in JSON
            query (function): it takes :class:`sqlalchemy.orm.query.Query`
            object as an argument, and returns in a modified form

                ::

                    def query(nodes):
                        return nodes.filter(node.__class__.tree_id.is_(node.tree_id))

                    node.get_tree(session=DBSession, json=True, query=query)

        Example:

        * :mod:`sqlalchemy_mptt.tests.cases.get_tree.test_get_tree`
        * :mod:`sqlalchemy_mptt.tests.cases.get_tree.test_get_json_tree`
        * :mod:`sqlalchemy_mptt.tests.cases.get_tree.test_get_json_tree_with_custom_field`
        """  # noqa
        tree = []
        nodes_of_level = {}

        # handle custom query
        nodes = cls._base_query(session)
        if query:
            nodes = query(nodes)
        nodes = cls._base_order(nodes).all()

        # search minimal level of nodes.
        min_level = min([node.level for node in nodes] or [None])

        def get_node_id(node):
            return getattr(node, node.get_pk_name())

        for node in nodes:
            result = cls._node_to_dict(node, json, json_fields)
            parent_id = node.parent_id
            if node.level != min_level:  # for cildren
                # Find parent in the tree
                if parent_id not in nodes_of_level.keys():
                    continue
                if 'children' not in nodes_of_level[parent_id]:
                    nodes_of_level[parent_id]['children'] = []
                # Append node to parent
                nl = nodes_of_level[parent_id]['children']
                nl.append(result)
                nodes_of_level[get_node_id(node)] = nl[-1]
            else:  # for top level nodes
                tree.append(result)
                nodes_of_level[get_node_id(node)] = tree[-1]
        return tree