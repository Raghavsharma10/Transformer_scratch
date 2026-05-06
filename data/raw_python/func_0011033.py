def _nonblocking_read(self):
        """Returns the number of characters read and adds them to self.unprocessed_bytes"""
        with Nonblocking(self.in_stream):
            if PY3:
                try:
                    data = os.read(self.in_stream.fileno(), READ_SIZE)
                except BlockingIOError:
                    return 0
                if data:
                    self.unprocessed_bytes.extend(data[i:i+1] for i in range(len(data)))
                    return len(data)
                else:
                    return 0
            else:
                try:
                    data = os.read(self.in_stream.fileno(), READ_SIZE)
                except OSError:
                    return 0
                else:
                    self.unprocessed_bytes.extend(data)
                    return len(data)