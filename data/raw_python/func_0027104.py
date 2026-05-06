def _did_fetch(self, connection):
        """ Fetching objects has been done """

        self.current_connection = connection
        response = connection.response
        should_commit = 'commit' not in connection.user_info or connection.user_info['commit']

        if connection.response.status_code >= 400 and BambouConfig._should_raise_bambou_http_error:
            raise BambouHTTPError(connection=connection)

        if response.status_code != 200:

            if should_commit:
                self.current_total_count = 0
                self.current_page = 0
                self.current_ordered_by = ''

            return self._send_content(content=None, connection=connection)

        results = response.data
        fetched_objects = list()
        current_ids = list()

        if should_commit:
            if 'X-Nuage-Count' in response.headers and response.headers['X-Nuage-Count']:
                self.current_total_count = int(response.headers['X-Nuage-Count'])

            if 'X-Nuage-Page' in response.headers and response.headers['X-Nuage-Page']:
                self.current_page = int(response.headers['X-Nuage-Page'])

            if 'X-Nuage-OrderBy' in response.headers and response.headers['X-Nuage-OrderBy']:
                self.current_ordered_by = response.headers['X-Nuage-OrderBy']

        if results:
            for result in results:
                nurest_object = self.new()
                nurest_object.from_dict(result)
                nurest_object.parent = self.parent_object

                fetched_objects.append(nurest_object)

                if not should_commit:
                    continue

                current_ids.append(nurest_object.id)

                if nurest_object in self:
                    idx = self.index(nurest_object)
                    current_object = self[idx]
                    current_object.from_dict(nurest_object.to_dict())
                else:
                    self.append(nurest_object)

            if should_commit:
                for obj in self:
                    if obj.id not in current_ids:
                        self.remove(obj)

        return self._send_content(content=fetched_objects, connection=connection)