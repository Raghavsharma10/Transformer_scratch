def applies_to(self, transition, from_state=None):
        """Whether this hook applies to the given transition/state.

        Args:
            transition (Transition): the transition to check
            from_state (State or None): the state to check. If absent, the check
                is 'might this hook apply to the related transition, given a
                valid source state'.
        """
        if '*' in self.names:
            return True
        elif self.kind in (HOOK_BEFORE, HOOK_AFTER, HOOK_CHECK):
            return self._match_transition(transition)
        elif self.kind == HOOK_ON_ENTER:
            return self._match_state(transition.target)
        elif from_state is None:
            # Testing whether the hook may apply to at least one source of the
            # transition
            return any(self._match_state(src) for src in transition.source)
        else:
            return self._match_state(from_state)