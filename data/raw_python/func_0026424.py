def _check_free_space(self):
        """Checks used filesystem storage sizes"""

        def get_folder_size(path):
            """Aggregates used size of a specified path, recursively"""

            total_size = 0
            for item in walk(path):
                for file in item[2]:
                    try:
                        total_size = total_size + getsize(join(item[0], file))
                    except (OSError, PermissionError) as e:
                        self.log("error with file:  " + join(item[0], file), e)
            return total_size

        for name, checkpoint in self.config.locations.items():
            try:
                stats = statvfs(checkpoint['location'])
            except (OSError, PermissionError) as e:
                self.log('Location unavailable:', name, e, type(e),
                         lvl=error, exc=True)
                continue
            free_space = stats.f_frsize * stats.f_bavail
            used_space = get_folder_size(
                checkpoint['location']
            ) / 1024.0 / 1024

            self.log('Location %s uses %.2f MB' % (name, used_space))

            if free_space < checkpoint['minimum']:
                self.log('Short of free space on %s: %.2f MB left' % (
                    name, free_space / 1024.0 / 1024 / 1024),
                         lvl=warn)