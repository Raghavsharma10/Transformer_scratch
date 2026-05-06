def adjust_for_isolated(self):
        """
        Remove certain plugins in order to handle the "isolated build"
        scenario.
        """
        if self.user_params.isolated.value:
            remove_plugins = [
                ("prebuild_plugins", "check_and_set_rebuild"),
                ("prebuild_plugins", "stop_autorebuild_if_disabled")
            ]

            for when, which in remove_plugins:
                self.pt.remove_plugin(when, which, 'removed from isolated build request')