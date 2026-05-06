def _post_transition(self, result, *args, **kwargs):
        """Performs post-transition actions."""
        for hook in self._filter_hooks(HOOK_AFTER, HOOK_ON_ENTER):
            hook(self.instance, result, *args, **kwargs)