def allocate_ip(self, hostipaddress, name, description):
        """
        Object method takes in input of hostipaddress, name and description and adds them to the
        parent ip scope.
        :param hostipaddress: str of ipv4 address of the target host ip record
        :param name: str of the name of the owner of the target host ip record
        :param description: str of a description of the target host ip record
        :return:
        """
        add_scope_ip(hostipaddress, name, description, self.auth, self.url, scopeid=self.id)