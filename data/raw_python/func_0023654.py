def check_finished(self):
        """Poll all processes and handle any finished processes."""
        changed = False
        for key in list(self.processes.keys()):
            # Poll process and check if it finshed
            process = self.processes[key]
            process.poll()
            if process.returncode is not None:
                # If a process is terminated by `stop` or `kill`
                # we want to queue it again instead closing it as failed.
                if key not in self.stopping:
                    # Get std_out and err_out
                    output, error_output = process.communicate()

                    descriptor = self.descriptors[key]
                    descriptor['stdout'].seek(0)
                    descriptor['stderr'].seek(0)
                    output = get_descriptor_output(descriptor['stdout'], key, handler=self)
                    error_output = get_descriptor_output(descriptor['stderr'], key, handler=self)

                    # Mark queue entry as finished and save returncode
                    self.queue[key]['returncode'] = process.returncode
                    if process.returncode != 0:
                        self.queue[key]['status'] = 'failed'
                    else:
                        self.queue[key]['status'] = 'done'

                    # Add outputs to queue
                    self.queue[key]['stdout'] = output
                    self.queue[key]['stderr'] = error_output
                    self.queue[key]['end'] = str(datetime.now().strftime("%H:%M"))

                    self.queue.write()
                    changed = True
                else:
                    self.stopping.remove(key)
                    if key in self.to_remove:
                        self.to_remove.remove(key)
                        del self.queue[key]
                    else:
                        if key in self.to_stash:
                            self.to_stash.remove(key)
                            self.queue[key]['status'] = 'stashed'
                        else:
                            self.queue[key]['status'] = 'queued'
                        self.queue[key]['start'] = ''
                        self.queue[key]['end'] = ''

                    self.queue.write()

                self.clean_descriptor(key)
                del self.processes[key]

        # If anything should be logged we return True
        return changed