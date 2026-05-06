def results(self):
        "Returns a dict of outputs from the GPTask execution."
        if self._results is None:
            results = self._json_struct['results']
            def result_iterator():
                for result in results:
                    datatype = None
                    conversion = None
                    for param in self.parent.parameters:
                        if param['name'] == result['paramName']:
                            datatype = param['datatype']
                    if datatype is None:
                        conversion = str
                    else:
                        conversion = datatype.fromJson
                    dt = result['paramName']
                    val = conversion(result['value'])
                    yield (dt, val)
            self._results = dict(res for res in result_iterator())
        return self._results