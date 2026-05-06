def gethosts(self):
        """
        Method gets all hosts currently allocated to the target scope and refreashes the self.hosts
        attributes of the object
        :return:
        """
        self.hosts = get_ip_scope_hosts(self.auth, self.url, self.id)