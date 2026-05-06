def get_descriptor(self, number):
        """Create file descriptors for process output."""
        # Create stdout file and get file descriptor
        stdout_path = os.path.join(self.config_dir,
                                   'pueue_process_{}.stdout'.format(number))
        if os.path.exists(stdout_path):
            os.remove(stdout_path)
        out_descriptor = open(stdout_path, 'w+')

        # Create stderr file and get file descriptor
        stderr_path = os.path.join(self.config_dir,
                                   'pueue_process_{}.stderr'.format(number))
        if os.path.exists(stderr_path):
            os.remove(stderr_path)
        err_descriptor = open(stderr_path, 'w+')

        self.descriptors[number] = {}
        self.descriptors[number]['stdout'] = out_descriptor
        self.descriptors[number]['stdout_path'] = stdout_path
        self.descriptors[number]['stderr'] = err_descriptor
        self.descriptors[number]['stderr_path'] = stderr_path
        return out_descriptor, err_descriptor