def activate(ctx):
        """Activate the given ParseContext."""
        if hasattr(ctx, '_on_context_exit'):
            raise ContextError(
                'Context actions registered outside this '
                'parse context are active')

        try:
            ParseContext._active.append(ctx)
            ctx._on_context_exit = []
            yield
        finally:
            for func, args, kwargs in ctx._on_context_exit:
                func(*args, **kwargs)
            del ctx._on_context_exit
            ParseContext._active.pop()