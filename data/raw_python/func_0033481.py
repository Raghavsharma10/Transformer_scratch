def share(self, with_user, roles=None, table=None):
        '''
        Give cube access rights to another user

        Not, this method is NOT supported by SQLite3!
        '''
        table = self.get_table(table)
        is_true(table is not None, 'invalid table: %s' % table)
        with_user = validate_username(with_user)
        roles = roles or ['SELECT']
        roles = validate_roles(roles, self.VALID_SHARE_ROLES)
        roles = list2str(roles)
        logger.info('Sharing cube %s with %s (%s)' % (table, with_user, roles))
        sql = 'GRANT %s ON %s TO %s' % (roles, table, with_user)
        return self.session_auto.execute(sql)