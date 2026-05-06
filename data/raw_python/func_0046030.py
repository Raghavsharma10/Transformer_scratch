def _get_next_object(self, object_class):
        """stub"""
        try:
            next_object = OsidList.next(self)
        except StopIteration:
            raise
        except Exception:  # Need to specify exceptions here!
            raise OperationFailed()
        if isinstance(next_object, dict):
            next_object = object_class(next_object)
        return next_object