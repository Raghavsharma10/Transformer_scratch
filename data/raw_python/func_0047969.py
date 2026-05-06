def find_files_for_tar(self, context, silent_build):
        """
        Return [(filename, arcname), ...] for all the files.
        """
        if not context.enabled:
            return

        files = self.find_files(context, silent_build)

        for path in files:
            relname = os.path.relpath(path, context.parent_dir)
            arcname = "./{0}".format(relname.encode('utf-8').decode('ascii', 'ignore'))
            if os.path.exists(path):
                yield path, arcname