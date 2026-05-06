def logs(self, build_id, follow=False, build_json=None, wait_if_missing=False):
        """
        provide logs from build

        :param build_id: str
        :param follow: bool, fetch logs as they come?
        :param build_json: dict, to save one get-build query
        :param wait_if_missing: bool, if build doesn't exist, wait
        :return: None, str or iterator
        """
        # does build exist?
        try:
            build_json = build_json or self.get_build(build_id).json()
        except OsbsResponseException as ex:
            if ex.status_code == 404:
                if not wait_if_missing:
                    raise OsbsException("Build '%s' doesn't exist." % build_id)
            else:
                raise

        if follow or wait_if_missing:
            build_json = self.wait_for_build_to_get_scheduled(build_id)

        br = BuildResponse(build_json)

        # When build is in new or pending state, openshift responds with 500
        if br.is_pending():
            return

        if follow:
            return self.stream_logs(build_id)

        buildlogs_url = self._build_url("builds/%s/log/" % build_id)
        response = self._get(buildlogs_url, headers={'Connection': 'close'})
        check_response(response)
        return response.content