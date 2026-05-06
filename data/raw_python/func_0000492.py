def cli_aliases(self):
        r"""Developer script aliases.
        """
        scripting_groups = []
        aliases = {}
        for cli_class in self.cli_classes:
            instance = cli_class()
            if getattr(instance, "alias", None):
                scripting_group = getattr(instance, "scripting_group", None)
                if scripting_group:
                    scripting_groups.append(scripting_group)
                    entry = (scripting_group, instance.alias)
                    if (scripting_group,) in aliases:
                        message = "alias conflict between scripting group"
                        message += " {!r} and {}"
                        message = message.format(
                            scripting_group, aliases[(scripting_group,)].__name__
                        )
                        raise Exception(message)
                    if entry in aliases:
                        message = "alias conflict between {} and {}"
                        message = message.format(
                            aliases[entry].__name__, cli_class.__name__
                        )
                        raise Exception(message)
                    aliases[entry] = cli_class
                else:
                    entry = (instance.alias,)
                    if entry in scripting_groups:
                        message = "alias conflict between {}"
                        message += " and scripting group {!r}"
                        message = message.format(cli_class.__name__, instance.alias)
                        raise Exception(message)
                    if entry in aliases:
                        message = "alias conflict be {} and {}"
                        message = message.format(cli_class.__name__, aliases[entry])
                        raise Exception(message)
                    aliases[(instance.alias,)] = cli_class
            else:
                if instance.program_name in scripting_groups:
                    message = "Alias conflict between {}"
                    message += " and scripting group {!r}"
                    message = message.format(cli_class.__name__, instance.program_name)
                    raise Exception(message)
                aliases[(instance.program_name,)] = cli_class
        alias_map = {}
        for key, value in aliases.items():
            if len(key) == 1:
                alias_map[key[0]] = value
            else:
                if key[0] not in alias_map:
                    alias_map[key[0]] = {}
                alias_map[key[0]][key[1]] = value
        return alias_map