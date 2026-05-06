def create_resource_quota(self, name, quota_json):
        """
        Prevent builds being scheduled and wait for running builds to finish.

        :return:
        """

        url = self._build_k8s_url("resourcequotas/")
        response = self._post(url, data=json.dumps(quota_json),
                              headers={"Content-Type": "application/json"})
        if response.status_code == http_client.CONFLICT:
            url = self._build_k8s_url("resourcequotas/%s" % name)
            response = self._put(url, data=json.dumps(quota_json),
                                 headers={"Content-Type": "application/json"})

        check_response(response)
        return response