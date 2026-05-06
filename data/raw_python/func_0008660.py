def cancel(self):
        """
        Cancel the consumer and stop recieving messages.

        This method is a :ref:`coroutine <coroutine>`.
        """
        self.sender.send_BasicCancel(self.tag)
        try:
            yield from self.synchroniser.wait(spec.BasicCancelOK)
        except AMQPError:
            pass
        else:
            # No need to call ready if channel closed.
            self.reader.ready()
        self.cancelled = True
        self.cancelled_future.set_result(self)
        if hasattr(self.callback, 'on_cancel'):
            self.callback.on_cancel()