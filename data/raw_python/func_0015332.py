def add_action_to(cls, parser, action, subactions, level):
        """Adds given action to given parser

        Args:
            parser: instance of devassistant_argparse.ArgumentParser
            action: devassistant.actions.Action subclass
            subactions: dict with subactions - {SubA: {SubB: {}}, SubC: {}}
        """
        p = parser.add_parser(action.name,
                              description=action.description,
                              argument_default=argparse.SUPPRESS)
        for arg in action.args:
            arg.add_argument_to(p)

        if subactions:
            subparsers = cls._add_subparsers_required(p,
                dest=settings.SUBASSISTANT_N_STRING.format(level),
                title=cls.subactions_str,
                description=cls.subactions_desc)
            for subact, subsubacts in sorted(subactions.items(), key=lambda x: x[0].name):
                cls.add_action_to(subparsers, subact, subsubacts, level + 1)