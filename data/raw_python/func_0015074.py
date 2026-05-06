def get_time_remaining_estimate(self):
        """
        Looks through all power sources and returns total time remaining estimate
        or TIME_REMAINING_UNLIMITED if ac power supply is online.
        """
        all_energy_now = []
        all_energy_not_discharging = []
        all_power_now = []
        for supply in os.listdir(POWER_SUPPLY_PATH):
            supply_path = os.path.join(POWER_SUPPLY_PATH, supply)
            try:
                type = self.power_source_type(supply_path)
                if type == common.POWER_TYPE_AC:
                    if self.is_ac_online(supply_path):
                        return common.TIME_REMAINING_UNLIMITED
                elif type == common.POWER_TYPE_BATTERY:
                    if self.is_battery_present(supply_path) and self.is_battery_discharging(supply_path):
                        energy_full, energy_now, power_now = self.get_battery_state(supply_path)
                        all_energy_now.append(energy_now)
                        all_power_now.append(power_now)
                    elif self.is_battery_present(supply_path) and not self.is_battery_discharging(supply_path):
                        energy_now = self.get_battery_state(supply_path)[1]
                        all_energy_not_discharging.append(energy_now)
                else:
                    warnings.warn("UPS is not supported.")
            except (RuntimeError, IOError) as e:
                warnings.warn("Unable to read properties of {0}: {1}".format(supply_path, e), category=RuntimeWarning)

        if len(all_energy_now) > 0:
            try:
                return sum([energy_now / power_now * 60.0 for energy_now, power_now in zip(all_energy_now, all_power_now)])\
                    + sum(all_energy_not_discharging) / (sum(all_power_now) / len(all_power_now)) * 60.0
            except ZeroDivisionError as e:
                warnings.warn("Unable to calculate time remaining estimate: {0}".format(e), category=RuntimeWarning)
                return common.TIME_REMAINING_UNKNOWN
        else:
            return common.TIME_REMAINING_UNKNOWN