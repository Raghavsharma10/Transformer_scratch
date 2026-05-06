def set_instance_erred(self, instance, error_message):
        """ Mark instance as erred and save error message """
        instance.set_erred()
        instance.error_message = error_message
        instance.save(update_fields=['state', 'error_message'])