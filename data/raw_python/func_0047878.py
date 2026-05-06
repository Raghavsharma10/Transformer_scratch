def extra_prepare(self, configuration, args_dict):
        """
        Called before the configuration.converters are activated

        Here we make sure that we have harpoon options from ``args_dict`` in
        the configuration.

        We then load all the harpoon modules as specified by the
        ``harpoon.addons`` setting.

        Finally we inject into the configuration:

        $@
            The ``harpoon.extra`` setting

        bash
            The ``bash`` setting

        command
            The ``command`` setting

        harpoon
            The harpoon settings

        collector
            This instance
        """
        harpoon = self.find_harpoon_options(configuration, args_dict)
        self.register = self.setup_addon_register(harpoon)

        # Make sure images is started
        if "images" not in self.configuration:
            self.configuration["images"] = {}

        # Add our special stuff to the configuration
        self.configuration.update(
            { "$@": harpoon.get("extra", "")
            , "bash": args_dict["bash"] or sb.NotSpecified
            , "harpoon": harpoon
            , "assume_role": args_dict["assume_role"] or NotSpecified
            , "command": args_dict['command'] or sb.NotSpecified
            , "collector": self
            }
        , source = "<args_dict>"
        )