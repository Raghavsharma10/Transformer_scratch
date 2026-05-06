def has_edge(self, node1_name, node2_name, account_for_direction=True):
        """ Proxies a call to the __has_edge method """
        return self.__has_edge(node1_name=node1_name, node2_name=node2_name, account_for_direction=account_for_direction)