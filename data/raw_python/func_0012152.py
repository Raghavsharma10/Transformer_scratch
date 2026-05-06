def publish_receiver_count(
            self, service, routing_id, method, timeout=None):
        '''Get the number of peers that would handle a particular publish

        This method will block until a response arrives

        :param service: the service name
        :type service: anything hash-able
        :param routing_id:
            the id used for narrowing within the service handlers
        :type routing_id: int
        :param method: the method name
        :type method: string
        :param timeout: maximum time to wait for the response
        :type timeout: int, float or None

        :raises:
            - :class:`Unroutable <junction.errors.Unroutable>` if no peers are
              registered to receive the message
            - :class:`WaitTimeout <junction.errors.WaitTimeout>` if a timeout
              was provided and it expires
        '''
        if not self._peer.up:
            raise errors.Unroutable()

        return self._rpc_client.recipient_count(self._peer,
                const.MSG_TYPE_PUBLISH, service, routing_id, method).wait(
                        timeout)[0]