def get_devices(self, condition=None, page_size=1000):
        """Iterates over each :class:`Device` for this device cloud account

        Examples::

            # get a list of all devices
            all_devices = list(dc.devicecore.get_devices())

            # build a mapping of devices by their vendor id using a
            # dict comprehension
            devices = dc.devicecore.get_devices()  # generator object
            devs_by_vendor_id = {d.get_vendor_id(): d for d in devices}

            # iterate over all devices in 'minnesota' group and
            # print the device mac and location
            for device in dc.get_devices(group_path == 'minnesota'):
                print "%s at %s" % (device.get_mac(), device.get_location())

        :param condition: An :class:`.Expression` which defines the condition
            which must be matched on the devicecore.  If unspecified,
            an iterator over all devices will be returned.
        :param int page_size: The number of results to fetch in a
            single page.  In general, the default will suffice.
        :returns: Iterator over each :class:`~Device` in this device cloud
            account in the form of a generator object.
        """

        condition = validate_type(condition, type(None), Expression, *six.string_types)
        page_size = validate_type(page_size, *six.integer_types)

        params = {"embed": "true"}
        if condition is not None:
            params["condition"] = condition.compile()

        for device_json in self._conn.iter_json_pages("/ws/DeviceCore", page_size=page_size, **params):
            yield Device(self._conn, self._sci, device_json)