def _eval_expectation(command, response, future):
        """Evaluate the response from Redis to see if it matches the expected
        response.

        :param command: The command that is being evaluated
        :type command: tredis.client.Command
        :param bytes response: The response value to check
        :param future: The future representing the execution of the command
        :type future: tornado.concurrent.Future
        :return:
        """
        if isinstance(command.expectation, int) and command.expectation > 1:
            future.set_result(response == command.expectation or response)
        else:
            future.set_result(response == command.expectation)