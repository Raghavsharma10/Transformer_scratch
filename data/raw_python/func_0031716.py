def receive(self, command_id, streams=('stdout', 'stderr'), command_timeout=60):
        """
        Recieves data
        :param command_id:
        :param streams:
        :param command_timeout:
        :return:
        """
        logging.info('receive command: ' + command_id)
        response_streams = dict.fromkeys(streams, '')
        (complete, exit_code) = self._receive_poll(command_id, response_streams)
        while not complete:
            (complete, exit_code) = self._receive_poll(command_id, response_streams)

        # This retains some compatibility with pywinrm
        if sorted(response_streams.keys()) == sorted(['stderr', 'stdout']):
            return response_streams['stdout'], response_streams['stderr'], exit_code
        else:
            return response_streams, exit_code