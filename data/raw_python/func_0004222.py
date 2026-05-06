def remove_connection(self, interface1, interface2):
        """Remove a connection between two interfaces"""

        uri = "api/interface/disconnect/%s/%s/" % (interface1, interface2)

        return self.delete(uri)