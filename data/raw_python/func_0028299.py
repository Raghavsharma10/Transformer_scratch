def addchild(self, startip, endip, name, description):
        """
        Method takes inpur of str startip, str endip, name, and description and adds a child scope.
        The startip and endip MUST be in the IP address range of the parent scope.
        :param startip: str of ipv4 address of the first address in the child scope
        :param endip: str of ipv4 address of the last address in the child scope
        :param name: of the owner of the child scope
        :param description: description of the child scope
        :return:
        """
        add_child_ip_scope(self.auth, self.url, startip, endip, name, description, self.id)