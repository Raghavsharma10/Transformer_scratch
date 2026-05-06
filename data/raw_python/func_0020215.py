def remove(self, value, session=None):
        '''Remove *value*, an instance of ``self.model`` from the set of
elements contained by the field.'''
        s, instance = self.session_instance('remove', value, session)
        # update state so that the instance does look persistent
        instance.get_state(iid=instance.pkvalue(), action='update')
        return s.delete(instance)