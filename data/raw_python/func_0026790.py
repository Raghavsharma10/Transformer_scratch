def handle_timer_event(self, handler):
        """
        Runs each timer handler in a separate greenlet thread.
        """
        while True:
            handler(self)
            sleep(handler.event.args["seconds"])