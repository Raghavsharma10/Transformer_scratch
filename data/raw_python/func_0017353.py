def collect(self, attrs):
        """Collect the implementations from a given attributes dict."""

        for name, value in attrs.items():
            if self.should_collect(value):
                transition = self.workflow.transitions[value.trname]

                if (
                        value.trname in self.implementations
                        and value.trname in self.custom_implems
                        and name != self.transitions_at[value.trname]):
                    # We already have an implementation registered.
                    other_implem_at = self.transitions_at[value.trname]
                    raise ValueError(
                        "Error for attribute %s: it defines implementation "
                        "%s for transition %s, which is already implemented "
                        "at %s." % (name, value, transition, other_implem_at))

                implem = self.add_implem(transition, name, value.func)
                self.custom_implems.add(transition.name)
                if value.check:
                    implem.add_hook(Hook(HOOK_CHECK, value.check))
                if value.before:
                    implem.add_hook(Hook(HOOK_BEFORE, value.before))
                if value.after:
                    implem.add_hook(Hook(HOOK_AFTER, value.after))