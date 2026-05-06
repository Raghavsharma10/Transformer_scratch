def init_command_set(self, scrapyd_url):
        """
         Initialize command set by scrapyd_url,each element is a list such as ['command','supported http method type']
        """

        if scrapyd_url[-1:] != '/':
            scrapyd_url = scrapyd_url + '/'
        self['daemonstatus'] = [scrapyd_url + 'daemonstatus.json', http_utils.METHOD_GET]
        self['addversion'] = [scrapyd_url + 'addversion.json', http_utils.METHOD_POST]
        self['schedule'] = [scrapyd_url + 'schedule.json', http_utils.METHOD_POST]
        self['cancel'] = [scrapyd_url + 'cancel.json', http_utils.METHOD_POST]
        self['listprojects'] = [scrapyd_url + 'listprojects.json', http_utils.METHOD_GET]
        self['listversions'] = [scrapyd_url + 'listversions.json', http_utils.METHOD_GET]
        self['listspiders'] = [scrapyd_url + 'listspiders.json', http_utils.METHOD_GET]
        self['listjobs'] = [scrapyd_url + 'listjobs.json', http_utils.METHOD_GET]
        self['delversion'] = [scrapyd_url + 'delversion.json', http_utils.METHOD_POST]
        self['delproject'] = [scrapyd_url + 'delproject.json', http_utils.METHOD_POST]
        self['logs'] = [scrapyd_url + 'logs/', http_utils.METHOD_GET]