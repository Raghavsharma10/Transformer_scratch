def get_providing_power_source_type(self):
        """
        Looks through all power supplies in POWER_SUPPLY_PATH.
        If there is an AC adapter online returns POWER_TYPE_AC.
        If there is a discharging battery, returns POWER_TYPE_BATTERY.
        Since the order of supplies is arbitrary, whatever found first is returned.
        """
        for supply in os.listdir(POWER_SUPPLY_PATH):
            supply_path = os.path.join(POWER_SUPPLY_PATH, supply)
            try:
                type = self.power_source_type(supply_path)
                if type == common.POWER_TYPE_AC:
                    if self.is_ac_online(supply_path):
                        return common.POWER_TYPE_AC
                elif type == common.POWER_TYPE_BATTERY:
                    if self.is_battery_present(supply_path) and self.is_battery_discharging(supply_path):
                        return common.POWER_TYPE_BATTERY
                else:
                    warnings.warn("UPS is not supported.")
            except (RuntimeError, IOError) as e:
                warnings.warn("Unable to read properties of {0}: {1}".format(supply_path, e), category=RuntimeWarning)

        return common.POWER_TYPE_AC