def api_post_multipart(self, action, data, binary_data_param):
        """
        Perform an HTTP Multipart POST request, using the shared-secret auth hash.
        @param action: API action call
        @param data: dictionary values
        @param: binary_data_params: array of multipart keys
        """
        binary_data = {}
        data = data.copy()

        try:
            file_handles = []
            for param in binary_data_param:
                if param in data:
                    binary_data[param] = file_handle = open(data[param], 'r')
                    file_handles.append(file_handle)
                    del data[param]
            json_payload = self._prepare_json_payload(data)

            return self._http_request(action, json_payload, "POST", binary_data)
        finally:
            for file_handle in file_handles:
                file_handle.close()