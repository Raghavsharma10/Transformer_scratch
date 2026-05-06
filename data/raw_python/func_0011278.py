def process_response(self, request, response, spider):
        """Overridden process_response would "pipe" response.body through BeautifulSoup."""
        return response.replace(body=str(BeautifulSoup(response.body, self.parser)))