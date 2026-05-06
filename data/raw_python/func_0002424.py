def get_fresh(self, columns=None):
        """
        Execute the query as a fresh "select" statement

        :param columns: The columns to get
        :type columns: list

        :return: The result
        :rtype: list
        """
        if not columns:
            columns = ['*']

        if not self.columns:
            self.columns = columns

        return self._processor.process_select(self, self._run_select())