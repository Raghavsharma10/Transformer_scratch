def path_to_root(self, session=None, order=desc):
        """Generate path from a leaf or intermediate node to the root.

        For example:

            node11.path_to_root()

            .. code::

                level           Nested sets example

                                 -----------------------------------------
                1               |    1(1)22                               |
                        ________|______|_____________________             |
                       |        |      |                     |            |
                       |         ------+---------            |            |
                2    2(2)5           6(4)11      | --     12(7)21         |
                       |               ^             |   /      \         |
                3    3(3)4       7(5)8   9(6)10      ---/----    \        |
                                                    13(8)16 |  17(10)20   |
                                                       |    |     |       |
                4                                   14(9)15 | 18(11)19    |
                                                            |             |
                                                             -------------
        """
        table = self.__class__
        query = self._base_query_obj(session=session)
        query = query.filter(table.is_ancestor_of(self, inclusive=True))
        return self._base_order(query, order=order)