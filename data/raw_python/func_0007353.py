def add_user_to_group(self, GroupID, UserID):
        """Add a user to a group."""
        # http://teampasswordmanager.com/docs/api-groups/#add_user
        log.info('Add User %s to Group %s' % (UserID, GroupID))
        self.put('groups/%s/add_user/%s.json' % (GroupID, UserID))