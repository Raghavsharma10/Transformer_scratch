def pubsub_pop_message(self, deadline=None):
        """Pops a message for a subscribed client.

        Args:
            deadline (int): max number of seconds to wait (None => no timeout)

        Returns:
            Future with the popped message as result (or None if timeout
                or ConnectionError object in case of connection errors
                or ClientError object if you are not subscribed)
        """
        if not self.subscribed:
            excep = ClientError("you must subscribe before using "
                                "pubsub_pop_message")
            raise tornado.gen.Return(excep)
        reply = None
        try:
            reply = self._reply_list.pop(0)
            raise tornado.gen.Return(reply)
        except IndexError:
            pass
        if deadline is not None:
            td = timedelta(seconds=deadline)
            yield self._condition.wait(timeout=td)
        else:
            yield self._condition.wait()
        try:
            reply = self._reply_list.pop(0)
        except IndexError:
            pass
        raise tornado.gen.Return(reply)