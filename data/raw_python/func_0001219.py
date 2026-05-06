def is_handler(cls, name, value):
        """Detect an handler and return its wanted signal name."""
        signal_name = False
        config = None
        if callable(value) and hasattr(value, SPEC_CONTAINER_MEMBER_NAME):
                spec = getattr(value, SPEC_CONTAINER_MEMBER_NAME)
                if spec['kind'] == 'handler':
                    signal_name = spec['name']
                    config = spec['config']
        return signal_name, config