def write(self, msg):
        """
        接收消息
        :param msg:
        :return:
        """

        self.app.events.before_response(self, msg)
        for bp in self.app.blueprints:
            bp.events.before_app_response(self, msg)

        try:
            self.child_output.put_nowait(msg)
            result = True
        except:
            logger.error('exc occur. msg: %r', msg, exc_info=True)
            result = False

        for bp in self.app.blueprints:
            bp.events.after_app_response(self, msg, result)
        self.app.events.after_response(self, msg, result)

        return result