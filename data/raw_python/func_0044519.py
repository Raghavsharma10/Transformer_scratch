def __processUsers(self):
        """Process users of the queue."""
        while self.__usersToProccess.empty() and not self.__end:
            pass

        while not self.__end or not self.__usersToProccess.empty():
            self.__lockGetUser.acquire()
            try:
                new_user = self.__usersToProccess.get(False)
            except Empty:
                self.__lockGetUser.release()
                return
            else:
                self.__lockGetUser.release()
                self.__addUser(new_user)
                self.__logger.info("__processUsers:" +
                                   str(self.__usersToProccess.qsize()) +
                                   " users to  process")