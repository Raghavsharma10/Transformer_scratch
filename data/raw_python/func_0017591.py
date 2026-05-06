def main(self):
        """
        A compulsary function that gets the output of the cmus-remote -Q command
        and converts it to unicode in order for it to be processed and finally
        output.
        """
        try:
            # Setting stderr to subprocess.STDOUT seems to stop the error
            # message returned by the process from being output to STDOUT.
            cmus_output = subprocess.check_output(['cmus-remote', '-Q'],
                                    stderr=subprocess.STDOUT).decode('utf-8')
        except subprocess.CalledProcessError:
            return self.output(None, None)
        if 'duration' in cmus_output:
            status = self.convert_cmus_output(cmus_output)
            out_string = self.options['format']
            for k, v in status.items():
                out_string = out_string.replace(k, v)
        else:
            out_string = None
        return self.output(out_string, out_string)