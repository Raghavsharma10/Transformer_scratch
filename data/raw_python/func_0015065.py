def get_providing_power_source_type(self):
        """
        Returns GetSystemPowerStatus().ACLineStatus

        @raise: WindowsError if any underlying error occures.
        """
        power_status = SYSTEM_POWER_STATUS()
        if not GetSystemPowerStatus(pointer(power_status)):
            raise WinError()
        return POWER_TYPE_MAP[power_status.ACLineStatus]