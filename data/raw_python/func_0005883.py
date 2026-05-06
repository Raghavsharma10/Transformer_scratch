def set_manage_params(
            self, chunked_input=None, chunked_output=None, gzip=None, websockets=None, source_method=None,
            rtsp=None, proxy_protocol=None):
        """Allows enabling various automatic management mechanics.

        * http://uwsgi.readthedocs.io/en/latest/Changelog-1.9.html#http-router-keepalive-auto-chunking-auto-gzip-and-transparent-websockets

        :param bool chunked_input: Automatically detect chunked input requests and put the session in raw mode.

        :param bool chunked_output: Automatically transform output to chunked encoding
            during HTTP 1.1 keepalive (if needed).

        :param bool gzip: Automatically gzip content if uWSGI-Encoding header is set to gzip,
            but content size (Content-Length/Transfer-Encoding) and Content-Encoding are not specified.

        :param bool websockets: Automatically detect websockets connections and put the session in raw mode.

        :param bool source_method: Automatically put the session in raw mode for `SOURCE` HTTP method.

            * http://uwsgi.readthedocs.io/en/latest/Changelog-2.0.5.html#icecast2-protocol-helpers

        :param bool rtsp: Allow the HTTP router to detect RTSP and chunked requests automatically.

        :param bool proxy_protocol: Allows the HTTP router to manage PROXY1 protocol requests,
            such as those made by Haproxy or Amazon Elastic Load Balancer (ELB).

        """
        self._set_aliased('chunked-input', chunked_input, cast=bool)
        self._set_aliased('auto-chunked', chunked_output, cast=bool)
        self._set_aliased('auto-gzip', gzip, cast=bool)
        self._set_aliased('websockets', websockets, cast=bool)
        self._set_aliased('manage-source', source_method, cast=bool)
        self._set_aliased('manage-rtsp', rtsp, cast=bool)
        self._set_aliased('enable-proxy-protocol', proxy_protocol, cast=bool)

        return self