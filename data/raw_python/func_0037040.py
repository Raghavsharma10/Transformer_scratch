def parse_instant_time(slot):
        """ Parse a slot into an InstantTime object.

        Sample response:

        {
          "entity": "snips/datetime",
          "range": {
            "end": 36,
            "start": 28
          },
          "rawValue": "tomorrow",
          "slotName": "weatherForecastStartDatetime",
          "value": {
            "grain": "Day",
            "kind": "InstantTime",
            "precision": "Exact",
            "value": "2017-09-15 00:00:00 +00:00"
          }
        }

        :param slot: a intent slot.
        :return: a parsed InstantTime object, or None.
        """
        date = IntentParser.get_dict_value(slot, ['value', 'value'])
        if not date:
            return None
        date = parse(date)
        if not date:
            return None
        grain = InstantTime.parse_grain(
            IntentParser.get_dict_value(slot,
                                        ['value', 'grain']))
        return InstantTime(date, grain)