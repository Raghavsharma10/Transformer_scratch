def instance_signals_and_handlers(cls, instance):
        """Calculate per-instance signals and handlers."""
        isignals = cls._signals.copy()

        ihandlers = cls._build_instance_handler_mapping(
            instance,
             cls._signal_handlers
        )
        return isignals, ihandlers