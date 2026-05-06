def get_logs(self, unique_id, logs, directory, pattern=constants.FILTER_NAME_ALLOW_NONE):
    """deprecated name for fetch_logs"""
    self.fetch_logs(unique_id, logs, directory, pattern)