def get_time_remaining_estimate(self):
        """
        In Mac OS X 10.7+
        Uses IOPSGetTimeRemainingEstimate to get time remaining estimate.

        In Mac OS X 10.6
        IOPSGetTimeRemainingEstimate is not available.
        If providing power source type is AC, returns TIME_REMAINING_UNLIMITED.
        Otherwise looks through all power sources returned by IOPSGetProvidingPowerSourceType
        and returns total estimate.
        """
        if IOPSGetTimeRemainingEstimate is not None: # Mac OS X 10.7+
            estimate = float(IOPSGetTimeRemainingEstimate())
            if estimate == -1.0:
                return common.TIME_REMAINING_UNKNOWN
            elif estimate == -2.0:
                return common.TIME_REMAINING_UNLIMITED
            else:
                return estimate / 60.0
        else: # Mac OS X 10.6
            warnings.warn("IOPSGetTimeRemainingEstimate is not preset", RuntimeWarning)
            blob = IOPSCopyPowerSourcesInfo()
            type = IOPSGetProvidingPowerSourceType(blob)
            if type == common.POWER_TYPE_AC:
                return common.TIME_REMAINING_UNLIMITED
            else:
                estimate = 0.0
                for source in IOPSCopyPowerSourcesList(blob):
                    description = IOPSGetPowerSourceDescription(blob, source)
                    if kIOPSIsPresentKey in description and description[kIOPSIsPresentKey] and kIOPSTimeToEmptyKey in description and description[kIOPSTimeToEmptyKey] > 0.0:
                        estimate += float(description[kIOPSTimeToEmptyKey])
                if estimate > 0.0:
                    return float(estimate)
                else:
                    return common.TIME_REMAINING_UNKNOWN