def handle_response(self, content, target=None, single_result=True, raw=False):
        """
        Parses response, checks it.
        """
        response = content['response']

        self.check_errors(response)

        data = response.get('data')

        if is_empty(data):
            return data
        elif is_paginated(data):
            if 'count' in data and not data['count']:
                # Response is paginated, but is empty
                return data['data']
            data = data['data']

        if raw:
            return data
        return self.init_all_objects(data, target=target, single_result=single_result)