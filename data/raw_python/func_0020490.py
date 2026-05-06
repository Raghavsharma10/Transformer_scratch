def create_build_config(self, build_config_json):
        """
        :return:
        """
        url = self._build_url("buildconfigs/")
        return self._post(url, data=build_config_json,
                          headers={"Content-Type": "application/json"})