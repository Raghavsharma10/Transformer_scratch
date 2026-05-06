def write(self):
        """Write the current queue to a file. We need this to continue an earlier session."""
        queue_path = os.path.join(self.config_dir, 'queue')
        queue_file = open(queue_path, 'wb+')
        try:
            pickle.dump(self.queue, queue_file, -1)
        except Exception:
            print('Error while writing to queue file. Wrong file permissions?')
        queue_file.close()