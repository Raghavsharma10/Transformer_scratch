def get_slot_value(payload, slot_name):
        """ Return the parsed value of a slot. An intent has the form:

            {
                "text": "brew me a cappuccino with 3 sugars tomorrow",
                "slots": [
                    {"value": {"slotName": "coffee_type", "value": "cappuccino"}},
                    ...
                ]
            }

            This function extracts a slot value given its slot name, and parses
            it into a Python object if applicable (e.g. for dates).

            Slots can be of various forms, the simplest being just:

            {"slotName": "coffee_sugar_amout", "value": "3"}

            More complex examples are date times, where we distinguish between
            instant times, or intervals. Thus, a slot:

            {
              "slotName": "weatherForecastStartDatetime",
              "value": {
                "kind": "InstantTime",
                "value": {
                  "value": "2017-07-14 00:00:00 +00:00",
                  "grain": "Day",
                  "precision": "Exact"
                }
              }
            }

            will be extracted as an `InstantTime` object, with datetime parsed
            and granularity set.

            Another example is a time interval:

            {
              "slotName": "weatherForecastStartDatetime",
              "value": {
                "kind": "TimeInterval",
                "value": {
                  "from": "2017-07-14 12:00:00 +00:00",
                  "to": "2017-07-14 19:00:00 +00:00"
                }
              },
            }

            which will be extracted as a TimeInterval object.

        :param payload: the intent, in JSON format.
        :return: the parsed value, as described above.
        """

        if not 'slots' in payload:
            return []

        slots = []
        for candidate in payload['slots']:
            if 'slotName' in candidate and candidate['slotName'] == slot_name:
                slots.append(candidate)

        result = []
        for slot in slots:
            kind = IntentParser.get_dict_value(slot, ['value', 'kind'])
            if kind == "InstantTime":
                result.append(IntentParser.parse_instant_time(slot))
            elif kind == "TimeInterval":
                result.append(IntentParser.parse_time_interval(slot))
            else:
                result.append(IntentParser.get_dict_value(slot, ['value', 'value', 'value']) \
                    or IntentParser.get_dict_value(slot, ['value', 'value']))

        return result