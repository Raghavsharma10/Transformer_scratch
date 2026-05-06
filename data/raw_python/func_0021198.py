def jscan(self, cursor=0, count=None, busyloop=None, queue=None,
              state=None, reply=None):
        """Iterate all the existing jobs in the local node.

        :param count: An hint about how much work to do per iteration.
        :param busyloop: Block and return all the elements in a busy loop.
        :param queue: Return only jobs in the specified queue.
        :param state: Must be a list - Return jobs in the specified state.
            Can be used multiple times for a logic OR.
        :param reply: None or string {"all", "id"} - Job reply type. Type can
            be all or id. Default is to report just the job ID. If all is
            specified the full job state is returned like for the SHOW command.
        """
        command = ["JSCAN", cursor]
        if count:
            command += ["COUNT", count]
        if busyloop:
            command += ["BUSYLOOP"]
        if queue:
            command += ["QUEUE", queue]
        if type(state) is list:
            for s in state:
                command += ["STATE", s]
        if reply:
            command += ["REPLY", reply]

        return self.execute_command(*command)