def cancel(self, project_name, job_id):
        """
        Cancel a spider run (aka. job). If the job is pending, it will be removed. If the job is running, it will be terminated.
        :param project_name: the project name
        :param job_id: the job id
        :return: a dictionary that status message
                 example: {"status": "ok", "prevstate": "running"}
        """
        url, method = self.command_set['cancel'][0], self.command_set['cancel'][1]
        data = {}
        data['project'] = project_name
        data['job'] = job_id
        response = http_utils.request(url, method_type=method, data=data, return_type=http_utils.RETURN_JSON)
        if response is None:
            logging.warning('%s failure: not found or connection fail' % sys._getframe().f_code.co_name)
            response = CancelResultSet().__dict__
        return response