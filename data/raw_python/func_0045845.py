def _notify_receiver(self, receiver, params, doc):
        """Send notification to the receiver"""
        verb = VMAP[doc['op']]
        ns = doc['ns']
        notification_id = Id(ns + 'Notification:' + str(ObjectId()) + '@' + params['authority'])
        object_id = Id(ns + ':' + str(doc['o']['_id']) + '@' + params['authority'])
        try:
            getattr(receiver, '_'.join([verb, params['obj_name_plural']]))(notification_id, [object_id])
        except AttributeError:
            pass
        return notification_id