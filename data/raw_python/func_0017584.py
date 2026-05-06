def get_news_aggregation(self):
        """
        Calling News Aggregation API

        Return:
           json data
        """

        news_aggregation_url = self.api_path + "news_aggregation" + "/"
        response = self.get_response(news_aggregation_url)
        return response