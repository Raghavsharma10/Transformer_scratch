def _execute_batch(self, actions):
        """
        Execute a single batch of Actions.
        For each action that has a problem, we annotate the action with the
        error information for that action, and we return the number of
        successful actions in the batch.
        :param actions: the list of Action objects to be executed
        :return: count of successful actions
        """
        wire_form = [a.wire_dict() for a in actions]
        if self.test_mode:
            result = self.make_call("/action/%s?testOnly=true" % self.org_id, wire_form)
        else:
            result = self.make_call("/action/%s" % self.org_id, wire_form)
        body = result.json()
        if body.get("errors", None) is None:
            if body.get("result") != "success":
                if self.logger: self.logger.warning("Server action result: no errors, but no success:\n%s", body)
            return len(actions)
        try:
            if body.get("result") == "success":
                if self.logger: self.logger.warning("Server action result: errors, but success report:\n%s", body)
            for error in body["errors"]:
                actions[error["index"]].report_command_error(error)
        except:
            raise ClientError(str(body), result)
        return body.get("completed", 0)