def get_matching_prefix(self, namespace, stream):
    """
    We look at the stream prefixs configured in stream.yaml and match stream
    to the longest prefix.
    """
    validate_stream(stream)
    default_prefix = ''
    longest_prefix = default_prefix
    for prefix in self.prefix_confs[namespace]:
      if prefix == default_prefix:
        continue
      if not stream.startswith(prefix):
        continue
      if len(prefix) <= len(longest_prefix):
        continue
      longest_prefix = prefix
    return longest_prefix