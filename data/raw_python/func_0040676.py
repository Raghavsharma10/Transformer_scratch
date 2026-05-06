def add_skip_ci_to_commit_msg(message: str) -> str:
        """
        Adds a "[skip ci]" tag at the end of a (possibly multi-line) commit message

        :param message: commit message
        :type message: str
        :return: edited commit message
        :rtype: str
        """
        first_line_index = message.find('\n')
        if first_line_index == -1:
            edited_message = message + ' [skip ci]'
        else:
            edited_message = message[:first_line_index] + ' [skip ci]' + message[first_line_index:]
        LOGGER.debug('edited commit message: %s', edited_message)
        return edited_message