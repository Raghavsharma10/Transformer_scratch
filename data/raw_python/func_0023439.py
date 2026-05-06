def read(self):
        """Read the queue of the last pueue session or set `self.queue = {}`."""
        queue_path = os.path.join(self.config_dir, 'queue')
        if os.path.exists(queue_path):
            queue_file = open(queue_path, 'rb')
            try:
                self.queue = pickle.load(queue_file)
            except Exception:
                print('Queue file corrupted, deleting old queue')
                os.remove(queue_path)
                self.queue = {}
            queue_file.close()
        else:
            self.queue = {}