def timeInfo(self):
        """Return the time info for this Map Service"""
        time_info = self._json_struct.get('timeInfo', {})
        if not time_info:
            return None
        time_info = time_info.copy()
        if 'timeExtent' in time_info:
            time_info['timeExtent'] = utils.timetopythonvalue(
                                                    time_info['timeExtent'])
        return time_info