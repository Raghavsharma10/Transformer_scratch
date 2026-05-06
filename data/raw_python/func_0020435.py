def has_ist_trigger(self):
        """Return True if this BuildConfig has ImageStreamTag trigger."""
        triggers = self.template['spec'].get('triggers', [])
        if not triggers:
            return False
        for trigger in triggers:
            if trigger['type'] == 'ImageChange' and \
                    trigger['imageChange']['from']['kind'] == 'ImageStreamTag':
                return True
        return False