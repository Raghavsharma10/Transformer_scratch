def put(self, measurementId):
        """
        Initiates a new measurement. Accepts a json payload with the following attributes;

        * duration: in seconds
        * startTime OR delay: a date in YMD_HMS format or a delay in seconds
        * description: some free text information about the measurement

        :return:
        """
        json = request.get_json()
        try:
            start = self._calculateStartTime(json)
        except ValueError:
            return 'invalid date format in request', 400
        duration = json['duration'] if 'duration' in json else 10
        if start is None:
            # should never happen but just in case
            return 'no start time', 400
        else:
            scheduled, message = self._measurementController.schedule(measurementId, duration, start,
                                                                      description=json.get('description'))
            return message, 200 if scheduled else 400