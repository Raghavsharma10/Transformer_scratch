def validate_object_action(self, action_name, obj=None):
        """ Execute validation for actions that are related to particular object """
        action_method = getattr(self, action_name)
        if not getattr(action_method, 'detail', False) and action_name not in ('update', 'partial_update', 'destroy'):
            # DRF does not add flag 'detail' to update and delete actions, however they execute operation with
            # particular object. We need to enable validation for them too.
            return
        validators = getattr(self, action_name + '_validators', [])
        for validator in validators:
            validator(obj or self.get_object())