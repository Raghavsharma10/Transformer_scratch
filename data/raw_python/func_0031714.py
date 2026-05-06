def open(self, input_streams=['stdin'], output_streams=['stderr', 'stdout']):
        """
        Opens the remote shell
        """
        shell = dict()
        shell['rsp:InputStreams'] = " ".join(input_streams)
        shell['rsp:OutputStreams'] = " ".join(output_streams)
        shell['rsp:IdleTimeout'] = str(self.idle_timeout)

        if self.working_directory is not None:
            shell['rsp:WorkingDirectory'] = str(self.working_directory)

        if self.environment is not None:
            variables = []
            for key, value in self.environment.items():
                variables.append({'#text': str(value), '@Name': key})
            shell['rsp:Environment'] = {'Variable': variables}

        response = self.session.create(self.resource, {'rsp:Shell': shell})
        self.__shell_id = response['rsp:Shell']['rsp:ShellId']