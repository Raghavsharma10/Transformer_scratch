def update_configuration(self, new_configuration):
        """ Save how much consumables were used and update current configuration.

            Return True if configuration changed.
        """
        if new_configuration == self.configuration:
            return False
        now = timezone.now()
        if now.month != self.price_estimate.month:
            raise ConsumptionDetailUpdateError('It is possible to update consumption details only for current month.')
        minutes_from_last_update = self._get_minutes_from_last_update(now)
        for consumable_item, usage in self.configuration.items():
            consumed_after_modification = usage * minutes_from_last_update
            self.consumed_before_update[consumable_item] = (
                self.consumed_before_update.get(consumable_item, 0) + consumed_after_modification)
        self.configuration = new_configuration
        self.last_update_time = now
        self.save()
        return True