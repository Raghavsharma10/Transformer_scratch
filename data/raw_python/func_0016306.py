def abort(self):
        """
        Handle request to cancel HTTP call
        """
        if (self.reply and self.reply.isRunning()):
            self.on_abort = True
            self.reply.abort()