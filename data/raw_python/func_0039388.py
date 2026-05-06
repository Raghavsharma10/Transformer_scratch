def response_builder(self, response):
        '''Try to return a pretty formatted response object
        '''
        try:
            r = response.json()
            result = r['query']['results']
            response = {
                'num_result': r['query']['count'] ,
                'result': result
            }
        except (Exception,) as e:
            print(e)
            return response.content

        return response