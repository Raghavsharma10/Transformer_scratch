def register_function_hooks(self, func):
        """Looks at an object method and registers it for relevent transitions."""
        for hook_kind, hooks in func.xworkflows_hook.items():
            for field_name, hook in hooks:
                if field_name and field_name != self.state_field:
                    continue
                for transition in self.workflow.transitions:
                    if hook.applies_to(transition):
                        implem = self.implementations[transition.name]
                        implem.add_hook(hook)