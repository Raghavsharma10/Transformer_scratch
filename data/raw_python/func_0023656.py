def spawn_new(self, key):
        """Spawn a new task and save it to the queue."""
        # Check if path exists
        if not os.path.exists(self.queue[key]['path']):
            self.queue[key]['status'] = 'failed'
            error_msg = "The directory for this command doesn't exist anymore: {}".format(self.queue[key]['path'])
            self.logger.error(error_msg)
            self.queue[key]['stdout'] = ''
            self.queue[key]['stderr'] = error_msg

        else:
            # Get file descriptors
            stdout, stderr = self.get_descriptor(key)

            if self.custom_shell != 'default':
                # Create subprocess
                self.processes[key] = subprocess.Popen(
                    [
                        self.custom_shell,
                        '-i',
                        '-c',
                        self.queue[key]['command'],
                    ],
                    stdout=stdout,
                    stderr=stderr,
                    stdin=subprocess.PIPE,
                    universal_newlines=True,
                    preexec_fn=os.setsid,
                    cwd=self.queue[key]['path']
                )
            else:
                # Create subprocess
                self.processes[key] = subprocess.Popen(
                    self.queue[key]['command'],
                    shell=True,
                    stdout=stdout,
                    stderr=stderr,
                    stdin=subprocess.PIPE,
                    universal_newlines=True,
                    preexec_fn=os.setsid,
                    cwd=self.queue[key]['path']
                )
            self.queue[key]['status'] = 'running'
            self.queue[key]['start'] = str(datetime.now().strftime("%H:%M"))

        self.queue.write()