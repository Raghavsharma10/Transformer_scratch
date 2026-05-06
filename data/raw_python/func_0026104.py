def offer(self, requestType, *args):
        """
        public interface to the reactor.
        :param requestType:
        :param args:
        :return:
        """
        if self._funcsByRequest.get(requestType) is not None:
            self._workQueue.put((requestType, list(*args)))
        else:
            logger.error("Ignoring unknown request on reactor " + self._name + " " + requestType)