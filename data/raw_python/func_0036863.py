def get_job_list(self, project_name):
        """
        Get the list of pending, running and finished jobs of some project.
        :param project_name: the project name
        :return: a dictionary that list inculde job name and status
                 example:
                 {"status": "ok",
                    "pending": [{"id": "78391cc0fcaf11e1b0090800272a6d06", "spider": "spider1"}],
                    "running": [{"id": "422e608f9f28cef127b3d5ef93fe9399", "spider": "spider2",
                    "start_time": "2012-09-12 10:14:03.594664"}],
                    "finished": [{"id": "2f16646cfcaf11e1b0090800272a6d06", "spider": "spider3",
                    "start_time": "2012-09-12 10:14:03.594664", "end_time": "2012-09-12 10:24:03.594664"}]}
        """
        url, method = self.command_set['listjobs'][0], self.command_set['listjobs'][1]
        data = {'project': project_name}
        response = http_utils.request(url, method_type=method, data=data, return_type=http_utils.RETURN_JSON)
        if response is None:
            logging.warning('%s failure: not found or connection fail' % sys._getframe().f_code.co_name)
            response = JobList().__dict__
        return response