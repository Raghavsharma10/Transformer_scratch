def validate(cls, cfg, path="", nested=0, parent_cfg=None):
        """
        Validates a section of a config dict. Will automatically
        validate child sections as well if their attribute pointers
        are instantiated with a handler property
        """

        # number of critical errors found
        num_crit = 0

        # number of non-critical errors found
        num_warn = 0


        # check for missing keys in the config
        for name in dir(cls):
            if nested > 0:
                break
            try:
                attr = getattr(cls, name)
                if isinstance(attr, Attribute):
                    if attr.default is None and name not in cfg:
                        # no default value defined, which means its required
                        # to be set in the config file
                        if path:
                            attr_full_name = "%s.%s" % (path, name)
                        else:
                            attr_full_name = name
                        raise vodka.exceptions.ConfigErrorMissing(
                            attr_full_name, attr)
                    attr.preload(cfg, name)

            except vodka.exceptions.ConfigErrorMissing as inst:
                if inst.level == "warn":
                    vodka.log.warn(inst.explanation)
                    num_warn += 1
                elif inst.level == "critical":
                    vodka.log.error(inst.explanation)
                    num_crit += 1


        if type(cfg) in [dict, Config]:
            keys = list(cfg.keys())
            if nested > 0:
                for _k, _v in cfg.items():
                    _num_crit, _num_warn = cls.validate(
                        _v,
                        path=("%s.%s" % (path, _k)),
                        nested=nested-1,
                        parent_cfg=cfg
                    )
                    num_crit += _num_crit
                    num_warn += _num_warn
                return num_crit, num_warn
        elif type(cfg) == list:
            keys = list(range(0, len(cfg)))
        else:
            raise ValueError("Cannot validate non-iterable config value")



        # validate existing keys in the config
        for key in keys:
            try:
                _num_crit, _num_warn = cls.check(cfg, key, path)
                num_crit += _num_crit
                num_warn += _num_warn
            except (
                vodka.exceptions.ConfigErrorUnknown,
                vodka.exceptions.ConfigErrorValue,
                vodka.exceptions.ConfigErrorType
            ) as inst:
                if inst.level == "warn":
                    vodka.log.warn(inst.explanation)
                    num_warn += 1
                elif inst.level == "critical":
                    vodka.log.error(inst.explanation)
                    num_crit += 1

        return num_crit, num_warn