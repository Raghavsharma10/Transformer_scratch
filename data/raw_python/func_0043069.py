def new_instance(type, frum, schema=None):
        """
        Factory!
        """
        if not type2container:
            _delayed_imports()

        if isinstance(frum, Container):
            return frum
        elif isinstance(frum, _Cube):
            return frum
        elif isinstance(frum, _Query):
            return _run(frum)
        elif is_many(frum):
            return _ListContainer(frum)
        elif is_text(frum):
            # USE DEFAULT STORAGE TO FIND Container
            if not config.default.settings:
                Log.error("expecting jx_base.container.config.default.settings to contain default elasticsearch connection info")

            settings = set_default(
                {
                    "index": join_field(split_field(frum)[:1:]),
                    "name": frum,
                },
                config.default.settings
            )
            settings.type = None  # WE DO NOT WANT TO INFLUENCE THE TYPE BECAUSE NONE IS IN THE frum STRING ANYWAY
            return type2container["elasticsearch"](settings)
        elif is_data(frum):
            frum = wrap(frum)
            if frum.type and type2container[frum.type]:
                return type2container[frum.type](frum.settings)
            elif frum["from"]:
                frum = copy(frum)
                frum["from"] = Container(frum["from"])
                return _Query.wrap(frum)
            else:
                Log.error("Do not know how to handle {{frum|json}}", frum=frum)
        else:
            Log.error("Do not know how to handle {{type}}", type=frum.__class__.__name__)