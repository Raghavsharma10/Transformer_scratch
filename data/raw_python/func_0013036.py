def blob_info(self):
    """Returns the BlobInfo for this file."""
    if not self.__blob_info:
      self.__blob_info = BlobInfo.get(self.__blob_key)
    return self.__blob_info