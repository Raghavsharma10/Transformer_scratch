def queue(self, name, value, quality=None, timestamp=None,
            attributes=None):
        """
        To reduce network traffic, you can buffer datapoints and
        then flush() anything in the queue.

        :param name: the name / label / tag for sensor data

        :param value: the sensor reading or value to record

        :param quality: the quality value, use the constants BAD, GOOD, etc.
            (optional and defaults to UNCERTAIN)

        :param timestamp: the time the reading was recorded in epoch
            milliseconds (optional and defaults to now)

        :param attributes: dictionary for any key-value pairs to store with the
            reading (optional)

        """
        # Get timestamp first in case delay opening websocket connection
        # and it must have millisecond accuracy
        if not timestamp:
            timestamp = int(round(time.time() * 1000))
        else:
            # Coerce datetime objects to epoch
            if isinstance(timestamp, datetime.datetime):
                timestamp = int(round(int(timestamp.strftime('%s')) * 1000))

        # Only specific quality values supported
        if quality not in [self.BAD, self.GOOD, self.NA, self.UNCERTAIN]:
            quality = self.UNCERTAIN

        # Check if adding to queue of an existing tag and add second datapoint
        for point in self._queue:
            if point['name'] == name:
                point['datapoints'].append([timestamp, value, quality])
                return

        # If adding new tag, initialize and set any attributes
        datapoint = {
            "name": name,
            "datapoints": [[timestamp, value, quality]]
        }

        # Attributes are extra details for a datapoint

        if attributes is not None:
            if not isinstance(attributes, dict):
                raise ValueError("Attributes are expected to be a dictionary.")

            # Validate rules for attribute keys to provide guidance.
            invalid_value = ':;= '
            has_invalid_value = re.compile(r'[%s]' % (invalid_value)).search
            has_valid_key = re.compile(r'^[\w\.\/\-]+$').search

            for (key, val) in list(attributes.items()):
                # Values cannot be empty
                if (val == '') or (val is None):
                    raise ValueError("Attribute (%s) must have a non-empty value." % (key))

                # Values should be treated as a string for regex validation
                val = str(val)

                # Values cannot contain certain arbitrary characters
                if bool(has_invalid_value(val)):
                    raise ValueError("Attribute (%s) cannot contain (%s)." %
                            (key, invalid_value))

                # Attributes have to be alphanumeric-ish
                if not bool(has_valid_key):
                    raise ValueError("Key (%s) not alphanumeric-ish." % (key))

            datapoint['attributes'] = attributes

        self._queue.append(datapoint)
        logging.debug("QUEUE: " + str(len(self._queue)))