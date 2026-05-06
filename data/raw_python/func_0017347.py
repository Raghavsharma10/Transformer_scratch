def _pre_transition_checks(self):
        """Run the pre-transition checks."""
        current_state = getattr(self.instance, self.field_name)
        if current_state not in self.transition.source:
            raise InvalidTransitionError(
                "Transition '%s' isn't available from state '%s'." %
                (self.transition.name, current_state.name))

        for check in self._filter_hooks(HOOK_CHECK):
            if not check(self.instance):
                raise ForbiddenTransition(
                    "Transition '%s' was forbidden by "
                    "custom pre-transition check." % self.transition.name)