def create_tcp_monitor(self, topics, batch_size=1, batch_duration=0,
                           compression='gzip', format_type='json'):
        """Creates a TCP Monitor instance in Device Cloud for a given list of topics

        :param topics: a string list of topics (e.g. ['DeviceCore[U]',
                  'FileDataCore']).
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
            <monTransportType>tcp</monTransportType>
            <monCompression>{compression}</monCompression>
        </Monitor>
        """.format(
            topics=','.join(topics),
            batch_size=batch_size,
            batch_duration=batch_duration,
            format_type=format_type,
            compression=compression,
        )
        monitor_xml = textwrap.dedent(monitor_xml)

        response = self._conn.post("/ws/Monitor", monitor_xml)
        location = ET.fromstring(response.text).find('.//location').text
        monitor_id = int(location.split('/')[-1])
        return TCPDeviceCloudMonitor(self._conn, monitor_id, self._tcp_client_manager)