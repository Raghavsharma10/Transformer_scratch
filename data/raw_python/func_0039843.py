def readline(self):
        """
        Waits for a line from the Herald client
        """
        content = {"session_id": self._session}
        prompt_msg = self._herald.send(
            self._peer, beans.Message(MSG_CLIENT_PROMPT, content))
        if prompt_msg.subject == MSG_SERVER_CLOSE:
            # Client closed its shell
            raise EOFError

        return prompt_msg.content