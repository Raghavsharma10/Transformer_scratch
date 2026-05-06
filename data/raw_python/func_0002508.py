def add_constraints(self):
        """
        Set the base constraints on the relation query.

        :rtype: None
        """
        parent_table = self._parent.get_table()

        self._set_join()

        if self._constraints:
            self._query.where('%s.%s' % (parent_table, self._first_key), '=', self._far_parent.get_key())