def _receive_poll(self, command_id, response_streams):
        """
        Recieves data
        :param command_id:
        :param streams:
        :return:
        """
        logging.info('receive command: ' + command_id)
        resource = ResourceLocator(CommandShell.ShellResource)
        resource.add_selector('ShellId', self.__shell_id)

        stream_attributes = {'#text': " ".join(response_streams.keys()), '@CommandId': command_id}
        receive = {'rsp:Receive': {'rsp:DesiredStream': stream_attributes}}

        try:
            response = self.session.recieve(resource, receive)['rsp:ReceiveResponse']
        except Exception as e:
            return False, None

        # some responses will not include any output
        session_streams = response.get('rsp:Stream', ())
        if not isinstance(session_streams, list):
            session_streams = [session_streams]

        for stream in session_streams:
            if stream['@CommandId'] == command_id and '#text' in stream:
                response_streams[stream['@Name']] += base64.b64decode(stream['#text'])
                # XPRESS Compression Testing
                # print "\\x".join("{:02x}".format(ord(c)) for c in base64.b64decode(stream['#text']))
                # data = base64.b64decode(stream['#text'])
                # f = open('c:\\users\\developer\\temp\\data.bin', 'wb')
                # f.write(data)
                # f.close()
                # decode = api.compression.xpress_decode(data[4:])
        done = response['rsp:CommandState']['@State'] == CommandShell.StateDone
        if done:
            exit_code = int(response['rsp:CommandState']['rsp:ExitCode'])
        else: exit_code = None
        return done, exit_code