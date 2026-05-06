def enable_asynchronous(self):
        """Check if socket have been monkey patched by gevent"""

        def is_monkey_patched():
            try:
                from gevent import monkey, socket
            except ImportError:
                return False
            if hasattr(monkey, "saved"):
                return "socket" in monkey.saved
            return gevent.socket.socket == socket.socket

        if not is_monkey_patched():
            raise Exception("To activate asynchonoucity, please monkey patch"
                            " the socket module with gevent")
        return True