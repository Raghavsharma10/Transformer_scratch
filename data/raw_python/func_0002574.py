def drilldown_tree(self, session=None, json=False, json_fields=None):
        """ This method generate a branch from a tree, begining with current
        node.

        For example:

            node7.drilldown_tree()

            .. code::

                level           Nested sets example
                1                    1(1)22       ---------------------
                        _______________|_________|_________            |
                       |               |         |         |           |
                2    2(2)5           6(4)11      |      12(7)21        |
                       |               ^         |         ^           |
                3    3(3)4       7(5)8   9(6)10  | 13(8)16   17(10)20  |
                                                 |    |          |     |
                4                                | 14(9)15   18(11)19  |
                                                 |                     |
                                                  ---------------------

        Example in tests:

            * :mod:`sqlalchemy_mptt.tests.cases.get_tree.test_drilldown_tree`
        """
        if not session:
            session = object_session(self)
        return self.get_tree(
            session,
            json=json,
            json_fields=json_fields,
            query=self._drilldown_query
        )