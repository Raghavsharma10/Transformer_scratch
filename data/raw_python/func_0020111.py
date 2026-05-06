def delete(self, instance, session):
        '''delete an *instance*'''
        if instance._meta.type == 'structure':
            return self._delete_structure(instance)
        inst = self.pop(instance)
        instance = inst if inst is not None else instance
        if instance is not None:
            state = instance.get_state()
            if state.persistent:
                state.deleted = True
                self._deleted[state.iid] = instance
                instance.session = session
            else:
                instance.session = None
            return instance