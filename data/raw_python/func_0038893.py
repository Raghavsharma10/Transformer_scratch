def SubmitJob(self, *params, **kw):
        """Asynchronously execute the specified GP task. This will return a 
           Geoprocessing Job object. Parameters are passed in either in order
           or as keywords."""
        fp = self.__expandparamstodict(params, kw)
        return self._get_subfolder('submitJob/', GPJob, fp)._jobstatus