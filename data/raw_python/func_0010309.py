def create_http_monitor(self, topics, transport_url, transport_token=None, transport_method='PUT', connect_timeout=0,
                            response_timeout=0, batch_size=1, batch_duration=0, compression='none', format_type='json'):
        """Creates a HTTP Monitor instance in Device Cloud for a given list of topics

        :param topics: a string list of topics (e.g. ['DeviceCore[U]',
                  'FileDataCore']).
        :param transport_url: URL of the customer web server.
        :param transport_token: Credentials for basic authentication in the following format: username:password
        :param transport_method: HTTP method to use for sending data: PUT or POST. The default is PUT.
        :param connect_timeout: A value of 0 means use the system default of 5000 (5 seconds).
        :param response_timeout: A value of 0 means use the system default of 5000 (5 seconds).
        :param batch_size: How many Msgs received before sending data.
        :param batch_duration: How long to wait before sending batch if it
            does not exceed batch_size.
        :param compression: Compression value (i.e. 'gzip').
        :param format_type: What format server should send data in (i.e. 'xml' or 'json').

        Returns an object of the created Monitor
        """

        monitor_xml = """\
        <Monitor>
            <monTopic>{topics}</monTopic>
            <monBatchSize>{batch_size}</monBatchSize>
            <monFormatType>{format_type}</monFormatType>
            <monTransportType>http</monTransportType>
            <monTransportUrl>{transport_url}</monTransportUrl>
            <monTransportToken>{transport_token}</monTransportToken>
            <monTransportMethod>{transport_method}</monTransportMethod>
            <monConnectTimeout>{connect_timeout}</monConnectTimeout>
            <monResponseTimeout>{response_timeout}</monResponseTimeout>
            <monCompression>{compression}</monCompression>
        </Monitor>
        """.format(
            topics=','.join(topics),
            transport_url=transport_url,
            transport_token=transport_token,
            transport_method=transport_method,
            connect_timeout=connect_timeout,
            response_timeout=response_timeout,
            batch_size=batch_size,
            batch_duration=batch_duration,
            format_type=format_type,
            compression=compression,
        )
        monitor_xml = textwrap.dedent(monitor_xml)

        response = self._conn.post("/ws/Monitor", monitor_xml)
        location = ET.fromstring(response.text).find('.//location').text
        monitor_id = int(location.split('/')[-1])
        return HTTPDeviceCloudMonitor(self._conn, monitor_id)