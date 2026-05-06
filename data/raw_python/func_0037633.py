def check(cls, cfg, key_name, path, parent_cfg=None):
        """
        Checks that the config values specified in key name is
        sane according to config attributes defined as properties
        on this class
        """

        attr = getattr(cls, key_name, None)

        if path != "":
            attr_full_name = "%s.%s" % (path, key_name)
        else:
            attr_full_name = key_name

        if not attr:
            # attribute specified by key_name is unknown, warn
            raise vodka.exceptions.ConfigErrorUnknown(attr_full_name)

        if attr.deprecated:
            vodka.log.warn("[config deprecated] %s is being deprecated in version %s" % (
                attr_full_name,
                attr.deprecated
            ))

        # prepare data
        for prepare in attr.prepare:
            cfg[key_name] = prepare(cfg[key_name])

        if hasattr(cls, "prepare_%s" % key_name):
            prepare = getattr(cls, "prepare_%s" % key_name)
            cfg[key_name] = prepare(cfg[key_name], config=cfg)

        value = cfg.get(key_name)

        if isinstance(attr.expected_type, types.FunctionType):
            # expected type holds a validator function
            p, reason = attr.expected_type(value)
            if not p:
                # validator did not pass
                raise vodka.exceptions.ConfigErrorValue(
                    attr_full_name,
                    attr,
                    value,
                    reason=reason
                )

        elif attr.expected_type != type(value):
            # attribute type mismatch
            raise vodka.exceptions.ConfigErrorType(
                attr_full_name,
                attr
            )

        if attr.choices and value not in attr.choices:
            # attribute value not valid according to
            # available choices
            raise vodka.exceptions.ConfigErrorValue(
                attr_full_name,
                attr,
                value
            )

        if hasattr(cls, "validate_%s" % key_name):
            # custom validator for this attribute was found
            validator = getattr(cls, "validate_%s" % key_name)
            valid, reason = validator(value)
            if not valid:
                # custom validator failed
                raise vodka.exceptions.ConfigErrorValue(
                    attr_full_name,
                    attr,
                    value,
                    reason=reason
                )

        num_crit = 0
        num_warn = 0


        if is_config_container(value) and attr.handler:
            if type(value) == dict or issubclass(type(value), Config):
                keys = list(value.keys())
            elif type(value) == list:
                keys = list(range(0, len(value)))
            else:
                return
            for k in keys:
                if not is_config_container(value[k]):
                    continue
                handler = attr.handler(k, value[k])
                if issubclass(handler, Handler):
                    h = handler
                else:
                    h = getattr(handler, "Configuration", None)

                #h = getattr(attr.handler(k, value[k]), "Configuration", None)
                if h:
                    if type(k) == int and type(value[k]) == dict and value[k].get("name"):
                        _path = "%s.%s" % (
                            attr_full_name, value[k].get("name"))
                    else:
                        _path = "%s.%s" % (attr_full_name, k)
                    _num_crit, _num_warn = h.validate(value[k], path=_path, nested=attr.nested, parent_cfg=cfg)
                    h.finalize(
                        value,
                        k,
                        value[k],
                        attr=attr,
                        attr_name=key_name,
                        parent_cfg=cfg
                    )
                    num_crit += _num_crit
                    num_warn += _num_warn

        attr.finalize(cfg, key_name, value, num_crit=num_crit)

        return (num_crit, num_warn)