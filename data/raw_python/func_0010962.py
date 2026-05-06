def on_status_withheld(self, status_id, user_id, countries):
        """Called when a status is withheld"""
        logger.info('Status %s withheld for user %s', status_id, user_id)
        return True