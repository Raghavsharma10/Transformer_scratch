def get_spider_list(self, project_name, version=None):
        """
        Get the list of spiders available in the last (unless overridden) version of some project.
        :param project_name: the project name
        :param version: the version of the project to examine
        :return: a dictionary that spider name list
                 example: {"status": "ok", "spiders": ["spider1", "spider2", "spider3"]}
        """
        url, method = self.command_set['listspiders'][0], self.command_set['listspiders'][1]
        data = {}
        data['project'] = project_name
        if version is not None:
            data['_version'] = version
        response = http_utils.request(url, method_type=method, data=data, return_type=http_utils.RETURN_JSON)
        if response is None:
            logging.warning('%s failure: not found or connection fail' % sys._getframe().f_code.co_name)
            response = SpiderList().__dict__
        return response