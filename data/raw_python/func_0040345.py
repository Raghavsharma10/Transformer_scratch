def get(self, note):
        """Resolve a single note into an object."""
        if self.closed:
            raise RuntimeError('{!r} already closed'.format(self))

        # Record request for note even if it fails to resolve.
        self.stats[note] += 1

        # Handle injection of partially applied annotated functions.
        if isinstance(note, tuple) and len(note) == 2:
            if note[0] == PARTIAL:
                fn, a, kw_items = note[1]
                return self.partial(fn, *a, **dict(kw_items))
            elif note[0] == PARTIAL_REGARDLESS:
                fn, a, kw_items = note[1]
                return self.partial_regardless(fn, *a, **dict(kw_items))
            elif note[0] == EAGER_PARTIAL:
                fn, a, kw_items = note[1]
                return self.eager_partial(fn, *a, **dict(kw_items))
            elif note[0] == EAGER_PARTIAL_REGARDLESS:
                fn, a, kw_items = note[1]
                return self.eager_partial_regardless(fn, *a, **dict(kw_items))

        basenote, name = self.parse_note(note)
        if name is None and basenote in self.values:
            return self.values[basenote]
        try:
            provider_factory = self.lookup(basenote)
        except LookupError:
            msg = "Unable to resolve '{}'"
            raise LookupError(msg.format(note))

        self.instantiating.append((basenote, name))
        try:
            if self.instantiating.count((basenote, name)) > 1:
                stack = ' <- '.join(repr(note) for note in self.instantiating)
                notes = tuple(self.instantiating)
                raise DependencyCycleError(stack, notes=notes)

            return self.handle_provider(provider_factory, note)
        finally:
            self.instantiating.pop()