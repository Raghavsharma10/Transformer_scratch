def loads(self, string):
    ''' load from a string '''
    for line in string.split("\n"):
      if len(line) < 10:
        continue

      try:
        report = ish_report()
        report.loads(line)
        self._reports.append(report)
      except BaseException as exp:
        ''' don't complain TOO much '''
        logging.warning('unable to load report, error: %s' % exp)