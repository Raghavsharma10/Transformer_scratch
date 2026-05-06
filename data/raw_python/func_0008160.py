def execute_single(self, action, immediate=False):
        """
        Execute a single action (containing commands on a single object).
        Normally, since actions are batched so as to be most efficient about execution,
        but if you want this command sent immediately (and all prior queued commands
        sent earlier in this command's batch), specify a True value for the immediate flag.

        Since any command can fill the current batch, your command may be submitted
        even if you don't specify the immediate flag.  So don't think of this always
        being a queue call if immedidate=False.

        Returns the number of actions in the queue, that got sent, and that executed successfully.

        :param action: the Action to be executed
        :param immediate: whether the Action should be executed immediately
        :return: the number of actions in the queue, that got sent, and that executed successfully.
        """
        return self.execute_multiple([action], immediate=immediate)