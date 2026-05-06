def _filter_hooks(self, *hook_kinds):
        """Filter a list of hooks, keeping only applicable ones."""
        hooks = sum((self.hooks.get(kind, []) for kind in hook_kinds), [])
        return sorted(hook for hook in hooks
                      if hook.applies_to(self.transition, self.current_state))