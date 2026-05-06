def context_spec(self):
        """Spec for specifying context options"""
        from harpoon.option_spec import image_objs
        return dict_from_bool_spec(lambda meta, val: {"enabled": val}
            , create_spec(image_objs.Context
                , validators.deprecated_key("use_git_timestamps", "Since docker 1.8, timestamps no longer invalidate the docker layer cache")

                , include = listof(string_spec())
                , exclude = listof(string_spec())
                , enabled = defaulted(boolean(), True)
                , find_options = string_spec()

                , parent_dir = directory_spec(formatted(defaulted(string_spec(), "{config_root}"), formatter=MergedOptionStringFormatter))
                , use_gitignore = defaulted(boolean(), False)
                , ignore_find_errors = defaulted(boolean(), False)
                )
            )