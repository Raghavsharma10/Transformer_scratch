def run_miner_if_free(self):
        """TODO: docstring"""
        (address, username, password, device, tstart, tend) = read_config()

        if self.dtype == 0:
            self.run_miner_cmd = [
                cpu_miner_path, '-o', address, '-O', '{}:{}'.format(
                    username, password)
            ]
        elif self.dtype == 1:
            # parse address -> scheme + netloc
            r = urlparse(address)

            # scheme://user[:password]@hostname:port
            url = '{}://{}:{}@{}'.format(r.scheme, username, password,
                                         r.netloc)

            # Cuda
            self.run_miner_cmd = [gpu_miner_path, '-P', url, '-U']

        if (len(self.run_miner_cmd) != 0):
            logger.info(' '.join(self.run_miner_cmd))

            # start if resource(cpu or gpu) is free
            if (self.is_device_free()):
                logger.info('start miner in another thread')
                self.run_cmd(self.run_miner_cmd)