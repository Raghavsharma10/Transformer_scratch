def _create_datadict(cls, internal_name):
        """Creates an object depending on `internal_name`

        Args:
            internal_name (str): IDD name

        Raises:
            ValueError: if `internal_name` cannot be matched to a data dictionary object

        """
        if internal_name == "LOCATION":
            return Location()
        if internal_name == "DESIGN CONDITIONS":
            return DesignConditions()
        if internal_name == "TYPICAL/EXTREME PERIODS":
            return TypicalOrExtremePeriods()
        if internal_name == "GROUND TEMPERATURES":
            return GroundTemperatures()
        if internal_name == "HOLIDAYS/DAYLIGHT SAVINGS":
            return HolidaysOrDaylightSavings()
        if internal_name == "COMMENTS 1":
            return Comments1()
        if internal_name == "COMMENTS 2":
            return Comments2()
        if internal_name == "DATA PERIODS":
            return DataPeriods()
        raise ValueError(
            "No DataDictionary known for {}".format(internal_name))