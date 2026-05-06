def extra_configuration_collection(self, configuration):
        """
        Hook to do any extra configuration collection or converter registration
        """
        harpoon_spec = HarpoonSpec()

        for image in configuration.get('images', {}).keys():
            self.make_image_converters(image, configuration, harpoon_spec)

        self.register_converters(
              { (0, ("content", )): sb.dictof(sb.string_spec(), sb.string_spec())
              , (0, ("harpoon", )): harpoon_spec.harpoon_spec
              , (0, ("authentication", )): harpoon_spec.authentications_spec
              }
            , Meta, configuration, sb.NotSpecified
            )

        # Some other code works better when harpoon no existy
        if configuration["harpoon"] is sb.NotSpecified:
            del configuration["harpoon"]