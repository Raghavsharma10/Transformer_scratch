def get_monitors(self, condition=None, page_size=1000):
        """Return an iterator over all monitors matching the provided condition

        Get all inactive monitors and print id::

            for mon in dc.monitor.get_monitors(MON_STATUS_ATTR == "DISABLED"):
                print(mon.get_id())

        Get all the HTTP monitors and print id::

            for mon in dc.monitor.get_monitors(MON_TRANSPORT_TYPE_ATTR == "http"):
                print(mon.get_id())

        Many other possibilities exist.  See the :mod:`devicecloud.condition` documention
        for additional details on building compound expressions.

        :param condition: An :class:`.Expression` which defines the condition
            which must be matched on the monitor that will be retrieved from
            Device Cloud. If a condition is unspecified, an iterator over
            all monitors for this account will be returned.
        :type condition: :class:`.Expression` or None
        :param int page_size: The number of results to fetch in a single page.
        :return: Generator yielding :class:`.DeviceCloudMonitor` instances matching the
            provided conditions.
        """
        req_kwargs = {}
        if condition:
            req_kwargs['condition'] = condition.compile()
        for monitor_data in self._conn.iter_json_pages("/ws/Monitor", **req_kwargs):
            yield DeviceCloudMonitor.from_json(self._conn, monitor_data, self._tcp_client_manager)