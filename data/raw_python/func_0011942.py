def accept_publish(
            self, service, mask, value, method, handler=None, schedule=False):
        '''Set a handler for incoming publish messages

        :param service: the incoming message must have this service
        :type service: anything hash-able
        :param mask:
            value to be bitwise-and'ed against the incoming id, the result of
            which must mask the 'value' param
        :type mask: int
        :param value:
            the result of `routing_id & mask` must match this in order to
            trigger the handler
        :type value: int
        :param method: the method name
        :type method: string
        :param handler:
            the function that will be called on incoming matching messages
        :type handler: callable
        :param schedule:
            whether to schedule a separate greenlet running ``handler`` for
            each matching message. default ``False``.
        :type schedule: bool

        :raises:
            - :class:`ImpossibleSubscription
              <junction.errors.ImpossibleSubscription>` if there is no routing
              ID which could possibly match the mask/value pair
            - :class:`OverlappingSubscription
              <junction.errors.OverlappingSubscription>` if a prior publish
              registration that overlaps with this one (there is a
              service/method/routing id that would match *both* this *and* a
              previously-made registration).
        '''
        # support @hub.accept_publish(serv, mask, val, meth) decorator usage
        if handler is None:
            return lambda h: self.accept_publish(
                    service, mask, value, method, h, schedule)

        log.info("accepting publishes%s %r" % (
                " scheduled" if schedule else "",
                (service, (mask, value), method),))

        self._dispatcher.add_local_subscription(const.MSG_TYPE_PUBLISH,
                service, mask, value, method, handler, schedule)

        return handler