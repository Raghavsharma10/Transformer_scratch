def _get_mpr_view(self, connection, table):
        """ Finds and returns view name in the sqlite db represented by given connection.

        Args:
            connection: connection to sqlite db where to look for partition table.
            table (orm.Table):

        Raises:
            MissingViewError: if database does not have partition table.

        Returns:
            str: database table storing partition data.

        """
        logger.debug(
            'Looking for view of the table.\n    table: {}'.format(table.vid))
        view = self.get_view_name(table)
        view_exists = self._relation_exists(connection, view)
        if view_exists:
            logger.debug(
                'View of the table exists.\n    table: {}, view: {}'
                .format(table.vid, view))
            return view
        raise MissingViewError('sqlite database does not have view for {} table.'
                               .format(table.vid))