def listen(self, *temporary_handlers):
        """When listen() is called all queued pygame.Events will be passed to all
        registered listeners. There are two ways to register a listener:

        1. as a permanent listener, that is always executed for every event. These
            are registered by passing the handler-functions during construction

        2. as a temporary listener, that will only be executed during the current
            call to listen(). These are registered by passing the handler functions
            as arguments to listen()

        When a handler is called it can provoke three different reactions through
        its return value.

        1. It can return EventConsumerInfo.DONT_CARE in which case the EventListener
            will pass the event to the next handler in line, or go to the next event,
            if the last handler was called.

        2. It can return EventConsumerInfo.CONSUMED in which case the event will not
            be passed to following handlers, and the next event in line will be
            processed.

        3. It can return anything else (including None, which will be returned if no
            return value is specified) in this case the listen()-method will return
            the result of the handler.

        Therefore all permanent handlers should usually return
        EventConsumerInfo.DONT_CARE
        """
        funcs = tuple(itt.chain(self.permanent_handlers, 
                          (proxy.listener for proxy in 
                            self.mouse_proxies[self.proxy_group].values()), 
                          temporary_handlers))

        for event in self._get_q():
            for func in funcs:
                ret = func(event)
                if ret == EventConsumerInfo.CONSUMED:
                    break
                if ret == EventConsumerInfo.DONT_CARE:
                    continue
                else:
                    return ret