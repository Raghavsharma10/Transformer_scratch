def add_configuration(self, configuration, collect_another_source, done, result, src):
        """
        Used to add a file to the configuration, result here is the yaml.load
        of the src.

        If the configuration we're reading in has ``harpoon.extra_files``
        then this is treated as a list of strings of other files to collect.

        We also take extra files to collector from result["images"]["__images_from__"]
        """
        # Make sure to maintain the original config_root
        if "config_root" in configuration:
            # if we already have a config root then we only keep new config root if it's not the home location
            # i.e. if it is the home configuration, we don't delete the new config_root
            if configuration["config_root"] != os.path.dirname(self.home_dir_configuration_location()):
                if "config_root" in result:
                    del result["config_root"]

        config_root = configuration.get("config_root")
        if config_root and src.startswith(config_root):
            src = "{{config_root}}/{0}".format(src[len(config_root) + 1:])

        if "images" in result and "__images_from__" in result["images"]:
            images_from_path = result["images"]["__images_from__"]

            if isinstance(images_from_path, six.string_types):
                images_from_path = [images_from_path]

            for ifp in images_from_path:

                if not ifp.startswith("/"):
                    ifp = os.path.join(os.path.dirname(src), ifp)

                if not os.path.exists(ifp) or not os.path.isdir(ifp):
                    raise self.BadConfigurationErrorKls(
                          "Specified folder for other configuration files points to a folder that doesn't exist"
                        , path="images.__images_from__"
                        , value=ifp
                        )

                for root, dirs, files in os.walk(ifp):
                    for fle in files:
                        location = os.path.join(root, fle)
                        if fle.endswith(".yml") or fle.endswith(".yaml"):
                            collect_another_source(location
                                , prefix = ["images", os.path.splitext(os.path.basename(fle))[0]]
                                )

            del result["images"]["__images_from__"]

        configuration.update(result, source=src)

        if "harpoon" in result:
            if "extra_files" in result["harpoon"]:
                spec = sb.listof(sb.formatted(sb.string_spec(), formatter=MergedOptionStringFormatter))
                config_root = {"config_root": result.get("config_root", configuration.get("config_root"))}
                meta = Meta(MergedOptions.using(result, config_root), []).at("harpoon").at("extra_files")
                for extra in spec.normalise(meta, result["harpoon"]["extra_files"]):
                    if os.path.abspath(extra) not in done:
                        if not os.path.exists(extra):
                            raise BadConfiguration("Specified extra file doesn't exist", extra=extra, source=src)
                        collect_another_source(extra)