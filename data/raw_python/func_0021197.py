def qscan(self, cursor=0, count=None, busyloop=None, minlen=None,
              maxlen=None, importrate=None):
        """
        Iterate all the existing queues in the local node.

        :param count: An hint about how much work to do per iteration.
        :param busyloop: Block and return all the elements in a busy loop.
        :param minlen: Don't return elements with less than count jobs queued.
        :param maxlen: Don't return elements with more than count jobs queued.
        :param importrate: Only return elements with an job import rate
                        (from other nodes) >= rate.
        """
        command = ["QSCAN", cursor]
        if count:
            command += ["COUNT", count]
        if busyloop:
            command += ["BUSYLOOP"]
        if minlen:
            command += ["MINLEN", minlen]
        if maxlen:
            command += ["MAXLEN", maxlen]
        if importrate:
            command += ["IMPORTRATE", importrate]

        return self.execute_command(*command)