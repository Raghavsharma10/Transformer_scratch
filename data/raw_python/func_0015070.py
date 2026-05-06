def get_providing_power_source_type(self):
        """
        Looks through all power supplies in POWER_SUPPLY_PATH.
        If there is an AC adapter online returns POWER_TYPE_AC.
        If there is a discharging battery, returns POWER_TYPE_BATTERY.
        Since the order of supplies is arbitrary, whatever found first is returned.
        """
        type = self.power_source_type()
        if type == common.POWER_TYPE_AC:
            if self.is_ac_online():
                return common.POWER_TYPE_AC
            elif type == common.POWER_TYPE_BATTERY:
                if self.is_battery_present() and self.is_battery_discharging():
                    return common.POWER_TYPE_BATTERY
                else:
                    warnings.warn("UPS is not supported.")
        return common.POWER_TYPE_AC