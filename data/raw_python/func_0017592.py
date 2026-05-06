def convert_cmus_output(self, cmus_output):
        """
        Change the newline separated string of output data into
        a dictionary which can then be used to replace the strings in the config
        format.

        cmus_output: A string with information about cmus that is newline
        seperated. Running cmus-remote -Q in a terminal will show you what
        you're dealing with.
        """
        cmus_output = cmus_output.split('\n')
        cmus_output = [x.replace('tag ', '') for x in cmus_output if not x in '']
        cmus_output = [x.replace('set ', '') for x in cmus_output]
        status = {}
        partitioned = (item.partition(' ') for item in cmus_output)
        status = {item[0]: item[2] for item in partitioned}
        status['duration'] = self.convert_time(status['duration'])
        status['position'] = self.convert_time(status['position'])
        return status