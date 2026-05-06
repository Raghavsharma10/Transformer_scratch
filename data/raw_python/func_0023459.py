def remove_old(self, max_log_time):
        """Remove all logs which are older than the specified time."""
        files = glob.glob('{}/queue-*'.format(self.log_dir))
        files = list(map(lambda x: os.path.basename(x), files))

        for log_file in files:
            # Get time stamp from filename
            name = os.path.splitext(log_file)[0]
            timestamp = name.split('-', maxsplit=1)[1]

            # Get datetime from time stamp
            time = datetime.strptime(timestamp, '%Y%m%d-%H%M')
            now = datetime.now()

            # Get total delta in seconds
            delta = now - time
            seconds = delta.total_seconds()

            # Delete log file, if the delta is bigger than the specified log time
            if seconds > int(max_log_time):
                log_filePath = os.path.join(self.log_dir, log_file)
                os.remove(log_filePath)