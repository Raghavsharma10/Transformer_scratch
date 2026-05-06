def get_low_battery_warning_level(self):
        """
        Looks through all power supplies in POWER_SUPPLY_PATH.
        If there is an AC adapter online returns POWER_TYPE_AC returns LOW_BATTERY_WARNING_NONE.
        Otherwise determines total percentage and time remaining across all attached batteries.
        """
        all_energy_full = []
        all_energy_now = []
        all_power_now = []
        try:
            type = self.power_source_type()
            if type == common.POWER_TYPE_AC:
                if self.is_ac_online():
                    return common.LOW_BATTERY_WARNING_NONE
            elif type == common.POWER_TYPE_BATTERY:
                if self.is_battery_present() and self.is_battery_discharging():
                    energy_full, energy_now, power_now = self.get_battery_state()
                    all_energy_full.append(energy_full)
                    all_energy_now.append(energy_now)
                    all_power_now.append(power_now)
            else:
                warnings.warn("UPS is not supported.")
        except (RuntimeError, IOError) as e:
            warnings.warn("Unable to read system power information!", category=RuntimeWarning)

        try:
            total_percentage = sum(all_energy_full) / sum(all_energy_now)
            total_time = sum([energy_now / power_now * 60.0 for energy_now, power_now in zip(all_energy_now, all_power_now)])
            if total_time <= 10.0:
                return common.LOW_BATTERY_WARNING_FINAL
            elif total_percentage <= 22.0:
                return common.LOW_BATTERY_WARNING_EARLY
            else:
                return common.LOW_BATTERY_WARNING_NONE
        except ZeroDivisionError as e:
            warnings.warn("Unable to calculate low battery level: {0}".format(e), category=RuntimeWarning)
            return common.LOW_BATTERY_WARNING_NONE