def get_observations(self):
    ''' return only specific weather observations (FM types) and
    ignore the summary of day reports '''
    return [rpt for rpt in self._reports if rpt.report_type in self.OBS_TYPES]