def ensure_cache_folder(self):
    """
    Creates a gradle cache folder if it does not exist.
    """
    if os.path.exists(self.cache_folder) is False:
      os.makedirs(self.cache_folder)