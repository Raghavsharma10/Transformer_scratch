def set_object_acl(self, obj):
        """ Set object ACL on creation if not already present. """
        if not obj._acl:
            from nefertari_guards import engine as guards_engine
            acl = self._factory(self.request).generate_item_acl(obj)
            obj._acl = guards_engine.ACLField.stringify_acl(acl)