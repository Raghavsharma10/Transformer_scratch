def backend_to_retrieve(self, namespace, stream):
    """
    Return backend enabled for reading for `stream`.
    """
    if namespace not in self.namespaces:
      raise NamespaceMissing('`{}` namespace is not configured'
                             .format(namespace))
    stream_prefix = self.get_matching_prefix(namespace, stream)
    read_backend = self.prefix_read_backends[namespace][stream_prefix]
    return (read_backend,
            self.prefix_confs[namespace][stream_prefix][read_backend])