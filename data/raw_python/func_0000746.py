def getitem_es(self, key):
        """ Override to support ACL filtering.

        To do so: passes `self.request` to `get_item` and uses
        `ACLFilterES`.
        """
        from nefertari_guards.elasticsearch import ACLFilterES
        es = ACLFilterES(self.item_model.__name__)
        params = {
            'id': key,
            'request': self.request,
        }
        obj = es.get_item(**params)
        obj.__acl__ = self.item_acl(obj)
        obj.__parent__ = self
        obj.__name__ = key
        return obj