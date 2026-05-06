def unsubscribe_rpc(self, service, mask, value):
        '''Remove a rpc subscription

        :param service: the service of the subscription to remove
        :type service: anything hash-able
        :param mask: the mask of the subscription to remove
        :type mask: int
        :param value: the value in the subscription to remove
        :type value: int
        :param handler: the handler function of the subscription to remove
        :type handler: callable

        :returns:
            a boolean indicating whether the subscription was there (True) and
            removed, or not (False)
        '''
        log.info("unsubscribing from RPC %r" % ((service, (mask, value)),))
        return self._dispatcher.remove_local_subscription(
                const.MSG_TYPE_RPC_REQUEST, service, mask, value)