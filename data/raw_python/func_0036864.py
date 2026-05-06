def get_logs(self, project_name, spider_name):
        """
        Get urls that scrapyd logs file by project name and spider name
        :param project_name: the project name
        :param spider_name: the spider name
        :return: two list of the logs file name and logs file url
        """
        url, method = self.command_set['logs'][0] + project_name + '/' + spider_name + '/', self.command_set['logs'][1]
        response = http_utils.request(url, method_type=method)
        html_parser = ScrapydLogsPageHTMLParser()
        html_parser.feed(response)
        html_parser.clean_enter_sign()
        return html_parser.result, [url + x for x in html_parser.result]