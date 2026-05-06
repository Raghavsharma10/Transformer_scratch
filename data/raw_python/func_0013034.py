def delete(self, **options):
    """Permanently delete this blob from Blobstore.

    Args:
      **options: Options for create_rpc().
    """
    fut = delete_async(self.key(), **options)
    fut.get_result()