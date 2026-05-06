def conditions(self, start, last_attempt):
        """
        Yield lines to execute in a docker context

        All conditions must evaluate for the container to be considered ready
        """
        if time.time() - start > self.timeout:
            yield WaitCondition.Timedout
            return

        if last_attempt is not None and time.time() - last_attempt < self.wait_between_attempts:
            yield WaitCondition.KeepWaiting
            return

        if self.greps is not NotSpecified:
            for name, val in self.greps.items():
                yield 'grep "{0}" "{1}"'.format(val, name)

        if self.file_value is not NotSpecified:
            for name, val in self.file_value.items():
                command = 'diff <(echo {0}) <(cat {1})'.format(val, name)
                if not self.harpoon.debug:
                    command = "{0} > /dev/null".format(command)
                yield command

        if self.port_open is not NotSpecified:
            for port in self.port_open:
                yield 'nc -z 127.0.0.1 {0}'.format(port)

        if self.curl_result is not NotSpecified:
            for url, content in self.curl_result.items():
                yield 'diff <(curl "{0}") <(echo {1})'.format(url, content)

        if self.file_exists is not NotSpecified:
            for path in self.file_exists:
                yield 'cat {0} > /dev/null'.format(path)

        if self.command not in (None, "", NotSpecified):
            for command in self.command:
                yield command