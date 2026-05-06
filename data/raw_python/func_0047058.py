def tasks_spec(self, available_actions, default_action="run"):
        """Tasks for a particular image"""
        return dictof(
              self.task_name_spec
            , create_spec(task_objs.Task, validators.deprecated_key("spec", "Use ``action`` and ``options`` instead (note that ``action`` defaults to run)")
                , action = defaulted(string_choice_spec(available_actions, "No such task"), default_action)
                , options = dictionary_spec()
                , overrides = dictionary_spec()
                , description = string_spec()
                )
            )