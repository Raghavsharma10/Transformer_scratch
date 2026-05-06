def _accept(self):
        """
        Work loop runs forever (or until running is False)
        :return:
        """
        logger.warning("Reactor " + self._name + " is starting")
        while self.running:
            try:
                self._completeTask()
            except:
                logger.exception("Unexpected exception during request processing")
        logger.warning("Reactor " + self._name + " is terminating")