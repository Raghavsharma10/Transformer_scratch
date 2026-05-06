def schedule(self,
                 project_name,
                 spider_name,
                 priority=0,
                 setting=None,
                 job_id=None,
                 version=None,
                 args={}):
        """
        Schedule a spider run (also known as a job), returning the job id.
        :param project_name: the project name
        :param spider_name: the spider name
        :param priority: the run priority
        :param setting: a Scrapy setting to use when running the spider
        :param job_id: a job id used to identify the job, overrides the default generated UUID
        :param version: the version of the project to use
        :param args: passed as spider argument
        :return: a dictionary that status message
                 example: {"status": "ok", "jobid": "6487ec79947edab326d6db28a2d86511e8247444"}
        """
        url, method = self.command_set['schedule'][0], self.command_set['schedule'][1]
        data = {}
        data['project'] = project_name
        data['spider'] = spider_name
        data['priority'] = priority
        if setting is not None:
            data['setting'] = setting
        if job_id is not None:
            data['jobid'] = job_id
        if version is not None:
            data['_version'] = version
        for k, v in args.items():
            data[k] = v
        response = http_utils.request(url, method_type=method, data=data, return_type=http_utils.RETURN_JSON)
        if response is None:
            logging.warning('%s failure: not found or connection fail' % sys._getframe().f_code.co_name)
            response = ScheduleResultSet().__dict__
        return response