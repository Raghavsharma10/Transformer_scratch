def delete_user_from_group(self, GroupID, UserID):
        """Delete a user from a group."""
        # http://teampasswordmanager.com/docs/api-groups/#del_user
        log.info('Delete user %s from group %s' % (UserID, GroupID))
        self.put('groups/%s/delete_user/%s.json' % (GroupID, UserID))