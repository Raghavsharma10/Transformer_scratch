def synchronize(self, user, info):
        '''
        It tries to do a group synchronization if possible
        This methods should be redeclared by the developer
        '''

        self.debug("Synchronize!")

        # Remove all groups from this user
        user.groups.clear()

        # For all domains found for this user
        for domain in info['groups']:
            # For all groups he is
            for groupname in info['groups'][domain]:
                # Lookup for that group
                group = Group.objects.filter(name=groupname).first()
                if group:
                    # If found, add the user to that group
                    user.groups.add(group)