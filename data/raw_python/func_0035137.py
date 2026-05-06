def send(self, data):
        """ Open transport, send data, and yield response chunks.
        """
        try:
            proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            raise URLError("Calling %r failed (%s)!" % (' '.join(self.cmd), exc))
        else:
            stdout, stderr = proc.communicate(data)
            if proc.returncode:
                raise URLError("Calling %r failed with RC=%d!\n%s" % (
                   ' '.join(self.cmd), proc.returncode, stderr,
                ))
            yield stdout