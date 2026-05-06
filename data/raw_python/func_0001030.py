def get_is_notification(self, obj):
        """ We say if the message should trigger a notification """
        try:
            o = compat_serializer_attr(self, obj)
            return o.is_notification
        except Exception:
            return False