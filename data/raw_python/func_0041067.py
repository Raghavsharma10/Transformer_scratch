def check(self, *exc_classes):
        """Check if any of ``exc_classes`` caused the failure.

        Arguments of this method can be exception types or type
        names (strings **fully qualified**). If captured exception is
        an instance of exception of given type, the corresponding argument
        is returned, otherwise ``None`` is returned.
        """
        for cls in exc_classes:
            cls_name = utils.cls_to_cls_name(cls)
            if cls_name in self._exc_type_names:
                return cls
        return None