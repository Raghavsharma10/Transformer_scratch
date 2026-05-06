def _calculateStartTime(self, json):
        """
        Calculates an absolute start time from the json payload. This is either the given absolute start time (+2s) or
        the time in delay seconds time. If the resulting date is in the past then now is returned instead.
        :param json: the payload from the UI
        :return: the absolute start time.
        """
        start = json['startTime'] if 'startTime' in json else None
        delay = json['delay'] if 'delay' in json else None
        if start is None and delay is None:
            return self._getAbsoluteTime(datetime.datetime.utcnow(), 2)
        elif start is not None:
            target = datetime.datetime.strptime(start, DATETIME_FORMAT)
            if target <= datetime.datetime.utcnow():
                time = self._getAbsoluteTime(datetime.datetime.utcnow(), 2)
                logger.warning('Date requested is in the past (' + start + '), defaulting to ' +
                               time.strftime(DATETIME_FORMAT))
                return time
            else:
                return target
        elif delay is not None:
            return self._getAbsoluteTime(datetime.datetime.utcnow(), delay)
        else:
            return None