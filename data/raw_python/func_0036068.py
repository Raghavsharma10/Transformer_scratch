def backends_to_mutate(self, namespace, stream):
    """
    Return all the backends enabled for writing for `stream`.
    """
    if namespace not in self.namespaces:
      raise NamespaceMissing('`{}` namespace is not configured'
                             .format(namespace))
    return self.prefix_confs[namespace][self.get_matching_prefix(namespace,
                                                                 stream)]