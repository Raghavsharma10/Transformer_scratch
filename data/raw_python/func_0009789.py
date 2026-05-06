def get_changed_devices(self, timestamp):
        """Get data since last timestamp.

        This is done via a blocking call, pass NONE for initial state.
        """
        if timestamp is None:
            payload = {}
        else:
            payload = {
                'timeout': SUBSCRIPTION_WAIT,
                'minimumdelay': SUBSCRIPTION_MIN_WAIT
            }
            payload.update(timestamp)
        # double the timeout here so requests doesn't timeout before vera
        payload.update({
            'id': 'lu_sdata',
        })

        logger.debug("get_changed_devices() requesting payload %s", str(payload))
        r = self.data_request(payload, TIMEOUT*2)
        r.raise_for_status()

        # If the Vera disconnects before writing a full response (as lu_sdata
        # will do when interrupted by a Luup reload), the requests module will
        # happily return 200 with an empty string. So, test for empty response,
        # so we don't rely on the JSON parser to throw an exception.
        if r.text == "":
            raise PyveraError("Empty response from Vera")

        # Catch a wide swath of what the JSON parser might throw, within
        # reason. Unfortunately, some parsers don't specifically return
        # json.decode.JSONDecodeError, but so far most seem to derive what
        # they do throw from ValueError, so that's helpful.
        try:
            result = r.json()
        except ValueError as ex:
            raise PyveraError("JSON decode error: " + str(ex))

        if not ( type(result) is dict
                 and 'loadtime' in result and 'dataversion' in result ):
            raise PyveraError("Unexpected/garbled response from Vera")

        # At this point, all good. Update timestamp and return change data.
        device_data = result.get('devices')
        timestamp = {
            'loadtime': result.get('loadtime'),
            'dataversion': result.get('dataversion')
        }
        return [device_data, timestamp]