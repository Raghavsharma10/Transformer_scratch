def harpoon_spec(self):
        """Spec for harpoon options"""
        formatted_string = formatted(string_spec(), MergedOptionStringFormatter, expected_type=six.string_types)
        formatted_boolean = formatted(boolean(), MergedOptionStringFormatter, expected_type=bool)

        return create_spec(Harpoon
            , config = optional_spec(file_spec())

            , tag = optional_spec(string_spec())
            , extra = defaulted(formatted_string, "")
            , debug = defaulted(boolean(), False)
            , addons = dictof(string_spec(), listof(string_spec()))
            , artifact = optional_spec(formatted_string)
            , extra_files = listof(string_spec())
            , chosen_task = defaulted(formatted_string, "list_tasks")
            , chosen_image = defaulted(formatted_string, "")

            , flat = defaulted(formatted_boolean, False)
            , no_cleanup = defaulted(formatted_boolean, False)
            , interactive = defaulted(formatted_boolean, True)
            , silent_build = defaulted(formatted_boolean, False)
            , keep_replaced = defaulted(formatted_boolean, False)
            , ignore_missing = defaulted(formatted_boolean, False)
            , no_intervention = defaulted(formatted_boolean, False)
            , intervene_afterwards = defaulted(formatted_boolean, False)

            , do_push = defaulted(formatted_boolean, False)
            , only_pushable = defaulted(formatted_boolean, False)
            , docker_context = any_spec()
            , docker_context_maker = any_spec()

            , stdout = defaulted(any_spec(), sys.stdout)
            , tty_stdin = defaulted(any_spec(), None)
            , tty_stdout = defaulted(any_spec(), lambda: sys.stdout)
            , tty_stderr = defaulted(any_spec(), lambda: sys.stderr)
            )