def rpc_receiver_count(self, service, routing_id):
        '''Get the number of peers that would handle a particular RPC

        :param service: the service name
        :type service: anything hash-able
        :param routing_id:
            the id used for narrowing within the service handlers
        :type routing_id: int

        :returns:
            the integer number of peers that would receive the described RPC
        '''
        peers = len(list(self._dispatcher.find_peer_routes(
            const.MSG_TYPE_RPC_REQUEST, service, routing_id)))
        if self._dispatcher.locally_handles(const.MSG_TYPE_RPC_REQUEST,
                service, routing_id):
            return peers + 1
        return peers