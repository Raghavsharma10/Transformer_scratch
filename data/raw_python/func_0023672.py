def _folder_get_content_iter(self, folder_key=None):
        """Iterator for api.folder_get_content"""

        lookup_params = [
            {'content_type': 'folders', 'node': 'folders'},
            {'content_type': 'files', 'node': 'files'}
        ]

        for param in lookup_params:
            more_chunks = True
            chunk = 0
            while more_chunks:
                chunk += 1
                content = self.api.folder_get_content(
                    content_type=param['content_type'], chunk=chunk,
                    folder_key=folder_key)['folder_content']

                # empty folder/file list
                if not content[param['node']]:
                    break

                # no next page
                if content['more_chunks'] == 'no':
                    more_chunks = False

                for resource_info in content[param['node']]:
                    yield resource_info