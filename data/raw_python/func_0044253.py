def get_holidays_for_year(self, types=None):
        """Get all the actual holiday days for a given HDate's year.

        If specified, use the list of types to limit the holidays returned.
        """
        # Filter any non-related holidays depending on Israel/Diaspora only
        holidays_list = [
            holiday for holiday in htables.HOLIDAYS if
            (holiday.israel_diaspora == "") or
            (holiday.israel_diaspora == "ISRAEL" and not self.diaspora) or
            (holiday.israel_diaspora == "DIASPORA" and self.diaspora)]

        if types:
            # Filter non-matching holiday types.
            holidays_list = [
                holiday for holiday in holidays_list if
                holiday.type in types
            ]

        # Filter any special cases defined by True/False functions
        holidays_list = [
            holiday for holiday in holidays_list if
            all(func(self) for func in holiday.date_functions_list)]

        def holiday_dates_cross_product(holiday):
            """Given a (days, months) pair, compute the cross product.

            If days and/or months are singletons, they are converted to a list.
            """
            return product(*([x] if isinstance(x, int) else x
                             for x in holiday.date))

        # Compute out every actual Hebrew date on which a holiday falls for
        # this year by exploding out the possible days for each holiday.
        holidays_list = [
            (holiday, HDate(
                heb_date=HebrewDate(self.hdate.year, date_instance[1],
                                    date_instance[0]),
                diaspora=self.diaspora,
                hebrew=self.hebrew))
            for holiday in holidays_list
            for date_instance in holiday_dates_cross_product(holiday)
            if len(holiday.date) >= 2
        ]
        return holidays_list