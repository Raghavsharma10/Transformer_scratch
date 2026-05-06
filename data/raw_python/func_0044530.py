def __launchThreads(self, numThreads):
        """Launch some threads and start to process users.

        :param numThreads: number of thrads to launch.
        :type numThreads: int.
        """
        i = 0
        while i < numThreads:
            self.__logger.debug("Launching thread number " +
                                str(i))
            i += 1
            newThr = Thread(target=self.__processUsers)
            newThr.setDaemon(True)
            self.__threads.add(newThr)
            newThr.start()