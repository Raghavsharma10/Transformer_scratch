def _run_event_methods(self, tag, stage=None):
        """Run code in the bundle that is marked with events. """
        import inspect
        from ambry.bundle.events import _runable_for_event

        funcs = []

        for func_name, f in inspect.getmembers(self, predicate=inspect.ismethod):
            if _runable_for_event(f, tag, stage):
                funcs.append(f)

        for func in funcs:
            func()