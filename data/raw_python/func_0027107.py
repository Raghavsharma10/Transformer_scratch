def _get_consumed(self, time):
        """ How many consumables were (or will be) used by resource until given time. """
        minutes_from_last_update = self._get_minutes_from_last_update(time)
        if minutes_from_last_update < 0:
            raise ConsumptionDetailCalculateError('Cannot calculate consumption if time < last modification date.')
        _consumed = {}
        for consumable_item in set(list(self.configuration.keys()) + list(self.consumed_before_update.keys())):
            after_update = self.configuration.get(consumable_item, 0) * minutes_from_last_update
            before_update = self.consumed_before_update.get(consumable_item, 0)
            _consumed[consumable_item] = after_update + before_update
        return _consumed