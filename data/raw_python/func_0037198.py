def _to_json(self):
        """Save :class:`~nmrstarlib.plsimulator.PeakList` into JSON string.

        :return: Peak list representation in JSON format.
        :rtype: :py:class:`str`
        """
        json_list = [{"Assignment": peak.assignments_list, "Dimensions": peak.chemshifts_list} for peak in self]
        return json.dumps(json_list, sort_keys=True, indent=4)