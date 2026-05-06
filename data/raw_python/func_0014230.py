def needs_compile(self):
        '''Returns True if self.sourcepath is newer than self.targetpath'''
        try:
            source_mtime = os.stat(self.sourcepath).st_mtime
        except OSError:  # no source for this template, so just return
            return False
        try:
            target_mtime = os.stat(self.targetpath).st_mtime
        except OSError: # target doesn't exist, so compile
            return True
        # both source and target exist, so compile if source newer
        return source_mtime > target_mtime