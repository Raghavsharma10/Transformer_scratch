def log_to_file(self, message):
        """Write a log message only to the file"""

        with self.build_fs.open(self.log_file, 'a+') as f:
            f.write(unicode(message + '\n'))