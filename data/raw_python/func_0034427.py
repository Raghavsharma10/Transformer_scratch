def _get_connection_from_url(self, url, timeout, **kwargs):
        """Returns a connection object given a string url"""

        url = self._decode_url(url, "")

        if url.scheme == 'http' or url.scheme == 'https':
            return HttpConnection(url.geturl(), timeout=timeout, **kwargs)
        else:
            if sys.version_info[0] > 2:
                raise ValueError("Thrift transport is not available "
                                 "for Python 3")

            try:
                from thrift_connection import ThriftConnection
            except ImportError:
                raise ImportError("The 'thrift' python package "
                                    "does not seem to be installed.")
            return ThriftConnection(url.hostname, url.port,
                                    timeout=timeout, **kwargs)