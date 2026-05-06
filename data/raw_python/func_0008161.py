def execute_multiple(self, actions, immediate=True):
        """
        Execute multiple Actions (each containing commands on a single object).
        Normally, the actions are sent for execution immediately (possibly preceded
        by earlier queued commands), but if you are going for maximum efficiency
        you can set immeediate=False which will cause the connection to wait
        and batch as many actions as allowed in each server call.

        Since any command can fill the current batch, one or more of your commands may be submitted
        even if you don't specify the immediate flag.  So don't think of this call as always
        being a queue call when immedidate=False.

        Returns the number of actions left in the queue, that got sent, and that executed successfully.

        NOTE: This is where we throttle the number of commands per action.  So the number
        of actions we were given may not be the same as the number we queue or send to the server.
        
        NOTE: If the server gives us a response we don't understand, we note that and continue
        processing as usual.  Then, at the end of the batch, we throw in order to warn the client
        that we had a problem understanding the server.

        :param actions: the list of Action objects to be executed
        :param immediate: whether to immediately send them to the server
        :return: tuple: the number of actions in the queue, that got sent, and that executed successfully.
        """
        # throttling part 1: split up each action into smaller actions, as needed
        # optionally split large lists of groups in add/remove commands (if action supports it)
        split_actions = []
        exceptions = []
        for a in actions:
            if len(a.commands) == 0:
                if self.logger: self.logger.warning("Sending action with no commands: %s", a.frame)
            # maybe_split_groups is a UserAction attribute, so the call may throw an AttributeError
            try:
                if a.maybe_split_groups(self.throttle_groups):
                    if self.logger: self.logger.debug(
                        "Throttling actions %s to have a maximum of %d entries in group lists.",
                        a.frame, self.throttle_groups)
            except AttributeError:
                pass
            if len(a.commands) > self.throttle_commands:
                if self.logger: self.logger.debug("Throttling action %s to have a maximum of %d commands.",
                                                  a.frame, self.throttle_commands)
                split_actions += a.split(self.throttle_commands)
            else:
                split_actions.append(a)
        actions = self.action_queue + split_actions
        # throttling part 2: execute the action list in batches, as needed
        sent = completed = 0
        batch_size = self.throttle_actions
        min_size = 1 if immediate else batch_size
        while len(actions) >= min_size:
            batch, actions = actions[0:batch_size], actions[batch_size:]
            if self.logger: self.logger.debug("Executing %d actions (%d remaining).", len(batch), len(actions))
            sent += len(batch)
            try:
                completed += self._execute_batch(batch)
            except Exception as e:
                exceptions.append(e)
        self.action_queue = actions
        self.local_status["actions-queued"] = queued = len(actions)
        self.local_status["actions-sent"] += sent
        self.local_status["actions-completed"] += completed
        if exceptions:
            raise BatchError(exceptions, queued, sent, completed)
        return queued, sent, completed