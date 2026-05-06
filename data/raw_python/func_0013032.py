def get_multi(cls, blob_keys, **ctx_options):
    """Multi-key version of get().

    Args:
      blob_keys: A list of blob keys.
      **ctx_options: Context options for Model().get_by_id().

    Returns:
      A list whose items are each either a BlobInfo entity or None.
    """
    futs = cls.get_multi_async(blob_keys, **ctx_options)
    return [fut.get_result() for fut in futs]