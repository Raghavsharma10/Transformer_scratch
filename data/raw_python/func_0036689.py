def get_extended_fsm_service(self):
        '''Get a reference to the ExtendedFsmService.

        @return A reference to the ExtendedFsmService object
        @raises InvalidSdoServiceError

        '''
        with self._mutex:
            try:
                return self._obj.get_sdo_service(RTC.ExtendedFsmService._NP_RepositoryId)._narrow(RTC.ExtendedFsmService)
            except:
                raise exceptions.InvalidSdoServiceError('ExtendedFsmService')