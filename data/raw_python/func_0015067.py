def get_time_remaining_estimate(self):
        """
        Returns time remaining estimate according to GetSystemPowerStatus().BatteryLifeTime
        """
        power_status = SYSTEM_POWER_STATUS()
        if not GetSystemPowerStatus(pointer(power_status)):
            raise WinError()

        if POWER_TYPE_MAP[power_status.ACLineStatus] == common.POWER_TYPE_AC:
            return common.TIME_REMAINING_UNLIMITED
        elif power_status.BatteryLifeTime == -1:
            return common.TIME_REMAINING_UNKNOWN
        else:
            return float(power_status.BatteryLifeTime) / 60.0