def setup_addon_register(self, harpoon):
        """Setup our addon register"""
        # Create the addon getter and register the crosshairs namespace
        self.addon_getter = AddonGetter()
        self.addon_getter.add_namespace("harpoon.crosshairs", Result.FieldSpec(), Addon.FieldSpec())

        # Initiate the addons from our configuration
        register = Register(self.addon_getter, self)

        if "addons" in harpoon:
            addons = harpoon["addons"]
            if type(addons) in (MergedOptions, dict) or getattr(addons, "is_dict", False):
                spec = sb.dictof(sb.string_spec(), sb.listof(sb.string_spec()))
                meta = Meta(harpoon, []).at("addons")
                for namespace, adns in spec.normalise(meta, addons).items():
                    register.add_pairs(*[(namespace, adn) for adn in adns])

        # Import our addons
        register.recursive_import_known()

        # Resolve our addons
        register.recursive_resolve_imported()

        return register