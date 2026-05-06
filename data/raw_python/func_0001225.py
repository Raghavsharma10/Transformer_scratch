def _get_class_handlers(cls, signal_name, instance):
        """Returns the handlers registered at class level.
        """
        handlers = cls._signal_handlers_sorted[signal_name]
        return [getattr(instance, hname) for hname in handlers]