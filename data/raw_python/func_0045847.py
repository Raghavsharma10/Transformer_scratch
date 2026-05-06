def _retry(self):
        """Deal with unacknowledged notifications."""
        notifications_to_delete = []
        for notification_id in self.notifications:
            if datetime.datetime.utcnow() > self.notifications[notification_id]['ts'] + self._wait_period:
                self._notify_receiver(
                    self.notifications[notification_id]['receiver'],
                    self.notifications[notification_id]['params'],
                    self.notifications[notification_id]['doc'])
                if self.notifications[notification_id]['attempts'] >= self._max_attempts - 1:
                    notifications_to_delete.append(notification_id)
                else:
                    self.notifications[notification_id]['ts'] = datetime.datetime.utcnow()
                    self.notifications[notification_id]['attempts'] += 1
        for notification_id in notifications_to_delete:
            del self.notifications[notification_id]