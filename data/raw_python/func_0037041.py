def parse_time_interval(slot):
        """ Parse a slot into a TimeInterval object.

        Sample response:

        {
          "entity": "snips/datetime",
          "range": {
            "end": 42,
            "start": 13
          },
          "rawValue": "between tomorrow and saturday",
          "slotName": "weatherForecastStartDatetime",
          "value": {
            "from": "2017-09-15 00:00:00 +00:00",
            "kind": "TimeInterval",
            "to": "2017-09-17 00:00:00 +00:00"
          }
        }

        :param slot: a intent slot.
        :return: a parsed TimeInterval object, or None.
        """
        start = IntentParser.get_dict_value(
            slot, ['value', 'from'])
        end = IntentParser.get_dict_value(slot, ['value', 'to'])
        if not start or not end:
            return None
        start = parse(start)
        end = parse(end)
        if not start or not end:
            return None
        return TimeInterval(start, end)