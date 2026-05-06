def capture_image(self, device_label):
        """ Capture smartcam image

        Args:
            device_label (str): device label of camera
        """
        response = None
        try:
            response = requests.post(
                urls.imagecapture(self._giid, device_label),
                headers={
                    'Content-Type': 'application/json',
                    'Cookie': 'vid={}'.format(self._vid)})
        except requests.exceptions.RequestException as ex:
            raise RequestError(ex)
        _validate_response(response)