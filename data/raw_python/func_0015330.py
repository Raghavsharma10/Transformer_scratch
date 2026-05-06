def generate_argument_parser(cls, tree, actions={}):
        """Generates argument parser for given assistant tree and actions.

        Args:
            tree: assistant tree as returned by
                  devassistant.assistant_base.AssistantBase.get_subassistant_tree
            actions: dict mapping actions (devassistant.actions.Action subclasses) to their
                     subaction dicts
        Returns:
            instance of devassistant_argparse.ArgumentParser (subclass of argparse.ArgumentParser)
        """
        cur_as, cur_subas = tree
        parser = devassistant_argparse.ArgumentParser(argument_default=argparse.SUPPRESS,
                                                      usage=argparse.SUPPRESS,
                                                      add_help=False)

        cls.add_default_arguments_to(parser)

        # add any arguments of the top assistant
        for arg in cur_as.args:
            arg.add_argument_to(parser)

        if cur_subas or actions:
            # then add the subassistants as arguments
            subparsers = cls._add_subparsers_required(parser,
                dest=settings.SUBASSISTANT_N_STRING.format('0'))
            for subas in sorted(cur_subas, key=lambda x: x[0].name):
                for alias in [subas[0].name] + getattr(subas[0], 'aliases', []):
                    cls.add_subassistants_to(subparsers, subas, level=1, alias=alias)

            for action, subactions in sorted(actions.items(), key=lambda x: x[0].name):
                cls.add_action_to(subparsers, action, subactions, level=1)

        return parser