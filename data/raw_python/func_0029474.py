def show_clock_output_clock_time_timezone(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        show_clock = ET.Element("show_clock")
        config = show_clock
        output = ET.SubElement(show_clock, "output")
        clock_time = ET.SubElement(output, "clock-time")
        timezone = ET.SubElement(clock_time, "timezone")
        timezone.text = kwargs.pop('timezone')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)