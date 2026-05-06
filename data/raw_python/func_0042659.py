def search_observations(self, search=None):
        """
        Search for observations, returning an Observation object for each result. FileRecords within result Observations
        have two additional methods patched into them, get_url() and download_to(file_name), which will retrieve the
        URL for the file content and download that content to a named file on disk, respectively.

        :param search:
            an instance of ObservationSearch - see the model docs for details on how to construct this
        :return:
            a dictionary containing 'count' and 'events'. 'events' is a sequence of Event objects containing the
            results of the search, and 'count' is the total number of results which would be returned if no result
            limit was in place (i.e. if the number of Events in the 'events' part is less than 'count' you have more
            records which weren't returned because of a query limit. Note that the default query limit is 100).
        """
        if search is None:
            search = model.ObservationSearch()
        search_string = _to_encoded_string(search)
        url = self.base_url + '/obs/{0}'.format(search_string)
        # print url
        response = requests.get(url)
        response_object = safe_load(response.text)
        obs_dicts = response_object['obs']
        obs_count = response_object['count']
        return {'count': obs_count,
                'events': [self._augment_observation_files(e)
                           for e in (model.Observation.from_dict(d)
                                     for d in obs_dicts)
                           ]
                }