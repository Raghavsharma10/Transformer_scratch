def unsubscribe_publish(self, service, mask, value):
        '''Remove a publish subscription

        :param service: the service of the subscription to remove
        :type service: anything hash-able
        :param mask: the mask of the subscription to remove
        :type mask: int
        :param value: the value in the subscription to remove
        :type value: int

        :returns:
            a boolean indicating whether the subscription was there (True) and
            removed, or not (False)
        '''
        log.info("unsubscribing from publish %r" % (
                (service, (mask, value)),))
        return self._dispatcher.remove_local_subscription(
                const.MSG_TYPE_PUBLISH, service, mask, value)