def get(cls, blob_key, **ctx_options):
    """Retrieve a BlobInfo by key.

    Args:
      blob_key: A blob key.  This may be a str, unicode or BlobKey instance.
      **ctx_options: Context options for Model().get_by_id().

    Returns:
      A BlobInfo entity associated with the provided key,  If there was
      no such entity, returns None.
    """
    fut = cls.get_async(blob_key, **ctx_options)
    return fut.get_result()