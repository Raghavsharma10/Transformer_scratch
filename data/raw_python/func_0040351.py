def register(cls, note, provider):
        """Implementation to register provider via `provider` & `factory`."""
        basenote, name = cls.parse_note(note)
        if 'provider_registry' not in vars(cls):
            cls.provider_registry = {}
        cls.provider_registry[basenote] = provider