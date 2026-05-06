def get_camera_imageseries(self, number_of_imageseries=10, offset=0):
        """ Get smartcam image series

        Args:
            number_of_imageseries (int): number of image series to get
            offset (int): skip offset amount of image series
        """
        response = None
        try:
            response = requests.get(
                urls.get_imageseries(self._giid),
                headers={
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Cookie': 'vid={}'.format(self._vid)},
                params={
                    "numberOfImageSeries": int(number_of_imageseries),
                    "offset": int(offset),
                    "fromDate": "",
                    "toDate": "",
                    "onlyNotViewed": "",
                    "_": self._giid})
        except requests.exceptions.RequestException as ex:
            raise RequestError(ex)
        _validate_response(response)
        return json.loads(response.text)