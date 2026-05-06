def open(self, print_matlab_welcome=False):
        '''Opens the matlab process.'''
        if self.process and not self.process.returncode:
            raise MatlabConnectionError('Matlab(TM) process is still active. Use close to '
                                            'close it')
        self.process = subprocess.Popen(
                [self.matlab_process_path, '-nojvm', '-nodesktop'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        flags = fcntl.fcntl(self.process.stdout, fcntl.F_GETFL)
        fcntl.fcntl(self.process.stdout, fcntl.F_SETFL, flags| os.O_NONBLOCK)

        if print_matlab_welcome:
            self._sync_output()
        else:
            self._sync_output(None)