def download_image(self, device_label, image_id, file_name):
        """ Download image taken by a smartcam

        Args:
            device_label (str): device label of camera
            image_id (str): image id from image series
            file_name (str): path to file
        """
        response = None
        try:
            response = requests.get(
                urls.download_image(self._giid, device_label, image_id),
                headers={
                    'Cookie': 'vid={}'.format(self._vid)},
                stream=True)
        except requests.exceptions.RequestException as ex:
            raise RequestError(ex)
        _validate_response(response)
        with open(file_name, 'wb') as image_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    image_file.write(chunk)