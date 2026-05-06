def get_low_battery_warning_level(self):
        """
        Returns warning according to GetSystemPowerStatus().BatteryLifeTime/BatteryLifePercent

        @raise WindowsError if any underlying error occures.
        """
        power_status = SYSTEM_POWER_STATUS()
        if not GetSystemPowerStatus(pointer(power_status)):
            raise WinError()

        if POWER_TYPE_MAP[power_status.ACLineStatus] == common.POWER_TYPE_AC:
            return common.LOW_BATTERY_WARNING_NONE
        else:
            if power_status.BatteryLifeTime != -1 and power_status.BatteryLifeTime <= 600:
                return common.LOW_BATTERY_WARNING_FINAL
            elif power_status.BatteryLifePercent <= 22:
                return common.LOW_BATTERY_WARNING_EARLY
            else:
                return common.LOW_BATTERY_WARNING_NONE