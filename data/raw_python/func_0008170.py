def report_command_error(self, error_dict):
        """
        Report a server error executing a command.

        We keep track of the command's position in the command list,
        and we add annotation of what the command was, to the error.
        :param error_dict: The server's error dict for the error encountered
        """
        error = dict(error_dict)
        error["command"] = self.commands[error_dict["step"]]
        error["target"] = self.frame
        del error["index"]  # throttling can change which action this was in the batch
        del error["step"]   # throttling can change which step this was in the action
        self.errors.append(error)