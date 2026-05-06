def image_spec(self):
        """Spec for each image"""
        from harpoon.option_spec import image_specs as specs
        from harpoon.option_spec import image_objs
        class persistence_shell_spec(Spec):
            """Make the persistence shell default to the shell on the image"""
            def normalise(self, meta, val):
                shell = defaulted(string_spec(), "/bin/bash").normalise(meta, meta.everything[["images", meta.key_names()["_key_name_2"]]].get("shell", NotSpecified))
                shell = defaulted(formatted(string_spec(), formatter=MergedOptionStringFormatter), shell).normalise(meta, val)
                return shell

        return create_spec(image_objs.Image
            , validators.deprecated_key("persistence", "The persistence feature has been removed")
            , validators.deprecated_key("squash_after", "The squash feature has been removed")
            , validators.deprecated_key("squash_before_push", "The squash feature has been removed")

            # Changed how volumes_from works
            , validators.deprecated_key("volumes_from", "Use ``volumes.share_with``")

            # Deprecated link
            , validators.deprecated_key("link", "Use ``links``")

            # Harpoon options
            , harpoon = any_spec()

            # default the name to the key of the image
            , tag = optional_spec(formatted(string_spec(), formatter=MergedOptionStringFormatter))
            , name = formatted(defaulted(string_spec(), "{_key_name_1}"), formatter=MergedOptionStringFormatter)
            , key_name = formatted(overridden("{_key_name_1}"), formatter=MergedOptionStringFormatter)
            , image_name = optional_spec(string_spec())
            , image_index = formatted(defaulted(string_spec(), ""), formatter=MergedOptionStringFormatter)
            , container_name = optional_spec(string_spec())
            , image_name_prefix = defaulted(string_spec(), "")

            , no_tty_option = defaulted(formatted(boolean(), formatter=MergedOptionStringFormatter), False)

            , user = defaulted(string_spec(), None)
            , configuration = any_spec()

            , vars = dictionary_spec()
            , assume_role = optional_spec(formatted(string_spec(), formatter=MergedOptionStringFormatter))
            , deleteable_image = defaulted(boolean(), False)

            , authentication = self.authentications_spec

            # The spec itself
            , shell = defaulted(formatted(string_spec(), formatter=MergedOptionStringFormatter), "/bin/bash")
            , bash = delayed(optional_spec(formatted(string_spec(), formatter=MergedOptionStringFormatter)))
            , command = delayed(optional_spec(formatted(string_spec(), formatter=MergedOptionStringFormatter)))
            , commands = required(container_spec(Commands, listof(command_spec())))
            , cache_from = delayed(or_spec(boolean(), listof(formatted(string_spec(), formatter=MergedOptionStringFormatter))))
            , cleanup_intermediate_images = defaulted(boolean(), True)

            , links = listof(specs.link_spec(), expect=image_objs.Link)

            , context = self.context_spec
            , wait_condition = optional_spec(self.wait_condition_spec)

            , lxc_conf = defaulted(filename_spec(), None)

            , volumes = create_spec(image_objs.Volumes
                , mount = listof(specs.mount_spec(), expect=image_objs.Mount)
                , share_with = listof(formatted(string_spec(), MergedOptionStringFormatter, expected_type=image_objs.Image))
                )

            , dependency_options = dictof(specs.image_name_spec()
                , create_spec(image_objs.DependencyOptions
                  , attached = defaulted(boolean(), False)
                  , wait_condition = optional_spec(self.wait_condition_spec)
                  )
                )

            , env = listof(specs.env_spec(), expect=image_objs.Environment)
            , ports = listof(specs.port_spec(), expect=image_objs.Port)
            , ulimits = defaulted(listof(dictionary_spec()), None)
            , log_config = defaulted(listof(dictionary_spec()), None)
            , security_opt = defaulted(listof(string_spec()), None)
            , read_only_rootfs = defaulted(boolean(), False)

            , other_options = create_spec(other_options
                , start = dictionary_spec()
                , build = dictionary_spec()
                , create = dictionary_spec()
                , host_config = dictionary_spec()
                )

            , network = create_spec(image_objs.Network
                , dns = defaulted(listof(string_spec()), None)
                , mode = defaulted(string_spec(), None)
                , hostname = defaulted(string_spec(), None)
                , domainname = defaulted(string_spec(), None)
                , disabled = defaulted(boolean(), False)
                , dns_search = defaulted(listof(string_spec()), None)
                , extra_hosts = listof(string_spec())
                , network_mode = defaulted(string_spec(), None)
                , publish_all_ports = defaulted(boolean(), False)
                )

            , cpu = create_spec(image_objs.Cpu
                , cap_add = defaulted(listof(string_spec()), None)
                , cpuset_cpus = defaulted(string_spec(), None)
                , cpuset_mems = defaulted(string_spec(), None)
                , cap_drop = defaulted(listof(string_spec()), None)
                , mem_limit = defaulted(integer_spec(), 0)
                , cpu_shares = defaulted(integer_spec(), None)
                , memswap_limit = defaulted(integer_spec(), 0)
                )

            , devices = defaulted(listof(dictionary_spec()), None)
            , privileged = defaulted(boolean(), False)
            , restart_policy = defaulted(string_spec(), None)
            )