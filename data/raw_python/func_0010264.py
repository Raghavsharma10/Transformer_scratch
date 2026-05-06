def get_groups(self, condition=None, page_size=1000):
        """Return an iterator over all groups in this device cloud account

        Optionally, a condition can be specified to limit the number of
        groups returned.

        Examples::

            # Get all groups and print information about them
            for group in dc.devicecore.get_groups():
                print group

            # Iterate over all devices which are in a group with a specific
            # ID.
            group = dc.devicore.get_groups(group_id == 123)[0]
            for device in dc.devicecore.get_devices(group_path == group.get_path()):
                print device.get_mac()

        :param condition: A condition to use when filtering the results set.  If
            unspecified, all groups will be returned.
        :param int page_size: The number of results to fetch in a
            single page.  In general, the default will suffice.
        :returns: Generator over the groups in this device cloud account.  No
            guarantees about the order of results is provided and child links
            between nodes will not be populated.

        """
        query_kwargs = {}
        if condition is not None:
            query_kwargs["condition"] = condition.compile()
        for group_data in self._conn.iter_json_pages("/ws/Group", page_size=page_size, **query_kwargs):
            yield Group.from_json(group_data)