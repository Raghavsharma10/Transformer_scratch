def check_trigger(self, date_tuple, utc_offset=0):
        """
        Returns boolean indicating if the trigger is active at the given time.
        The date tuple should be in the local time. Unless periodicities are
        used, utc_offset does not need to be specified. If periodicities are
        used, specifically in the hour and minutes fields, it is crucial that
        the utc_offset is specified.
        """
        year, month, day, hour, mins = date_tuple
        given_date = datetime.date(year, month, day)
        zeroday = datetime.date(*self.epoch[:3])
        last_dom = calendar.monthrange(year, month)[-1]
        dom_matched = True

        # In calendar and datetime.date.weekday, Monday = 0
        given_dow = (datetime.date.weekday(given_date) + 1) % 7
        first_dow = (given_dow + 1 - day) % 7

        # Figure out how much time has passed from the epoch to the given date
        utc_diff = utc_offset - self.epoch[5]
        mod_delta_yrs = year - self.epoch[0]
        mod_delta_mon = month - self.epoch[1] + mod_delta_yrs * 12
        mod_delta_day = (given_date - zeroday).days
        mod_delta_hrs = hour - self.epoch[3] + mod_delta_day * 24 + utc_diff
        mod_delta_min = mins - self.epoch[4] + mod_delta_hrs * 60

        # Makes iterating through like components easier.
        quintuple = zip(
            (mins, hour, day, month, given_dow),
            self.numerical_tab,
            self.string_tab,
            (mod_delta_min, mod_delta_hrs, mod_delta_day, mod_delta_mon,
                mod_delta_day),
            FIELD_RANGES)

        for value, valid_values, field_str, delta_t, field_type in quintuple:
            # All valid, static values for the fields are stored in sets
            if value in valid_values:
                continue

            # The following for loop implements the logic for context
            # sensitive and epoch sensitive constraints. break statements,
            # which are executed when a match is found, lead to a continue
            # in the outer loop. If there are no matches found, the given date
            # does not match expression constraints, so the function returns
            # False as seen at the end of this for...else... construct.
            for cron_atom in field_str.split(','):
                if cron_atom[0] == '%':
                    if not(delta_t % int(cron_atom[1:])):
                        break

                elif '#' in cron_atom:
                    D, N = int(cron_atom[0]), int(cron_atom[2])
                    # Computes Nth occurence of D day of the week
                    if (((D - first_dow) % 7) + 1 + 7 * (N - 1)) == day:
                        break

                elif cron_atom[-1] == 'W':
                    target = min(int(cron_atom[:-1]), last_dom)
                    lands_on = (first_dow + target - 1) % 7
                    if lands_on == 0:
                        # Shift from Sun. to Mon. unless Mon. is next month
                        if target < last_dom:
                            target += 1
                        else:
                            target -= 2
                    elif lands_on == 6:
                        # Shift from Sat. to Fri. unless Fri. in prior month
                        if target > 1:
                            target -= 1
                        else:
                            target += 2

                    # Break if the day is correct, and target is a weekday
                    if target == day and (first_dow + target) % 7 > 1:
                        break

                elif cron_atom[-1] == 'L':
                    # In dom field, L means the last day of the month
                    target = last_dom

                    if field_type == DAYS_OF_WEEK:
                        # Calculates the last occurence of given day of week
                        desired_dow = int(cron_atom[:-1])
                        target = (((desired_dow - first_dow) % 7) + 29)
                        if target > last_dom:
                            target -= 7

                    if target == day:
                        break
            else:
                # See 2010.11.15 of CHANGELOG
                if field_type == DAYS_OF_MONTH and self.string_tab[4] != '*':
                    dom_matched = False
                    continue
                elif field_type == DAYS_OF_WEEK and self.string_tab[2] != '*':
                    # If we got here, then days of months validated so it does
                    # not matter that days of the week failed.
                    return dom_matched

                # None of the expressions matched which means this field fails
                return False

        # Arriving at this point means the date landed within the constraints
        # of all fields; the associated trigger should be fired.
        return True