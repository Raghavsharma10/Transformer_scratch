def get_observatory_status(self, observatory_id, status_time=None):
        """
        Get details of the specified camera's status

        :param string observatory_id:
            a observatory ID, as returned by list_observatories()
        :param float status_time:
            optional, if specified attempts to get the status for the given camera at a particular point in time
            specified as a datetime instance. This is useful if you want to retrieve the status of the camera at the
            time a given event or file was produced. If this is None or not specified the time is 'now'.
        :return:
            a dictionary, or None if there was either no observatory found.
        """
        if status_time is None:
            response = requests.get(
                self.base_url + '/obstory/{0}/statusdict'.format(observatory_id))
        else:
            response = requests.get(
                self.base_url + '/obstory/{0}/statusdict/{1}'.format(observatory_id, str(status_time)))
        if response.status_code == 200:
            d = safe_load(response.text)
            if 'status' in d:
                return d['status']
        return None