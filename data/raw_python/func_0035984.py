def _prevent_core_dump(cls):
        """Prevent the process from generating a core dump."""
        try:
            # Try to get the current limit
            resource.getrlimit(resource.RLIMIT_CORE)
        except ValueError:
            # System doesn't support the RLIMIT_CORE resource limit
            return
        else:
            # Set the soft and hard limits for core dump size to zero
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))