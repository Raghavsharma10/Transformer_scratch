def _set_config(self, config=None):
        """Set this component's initial configuration"""
        if not config:
            config = {}

        try:
            # pprint(self.configschema)
            self.config = self.componentmodel(config)
            # self.log("Config schema:", lvl=critical)
            # pprint(self.config.__dict__)

            # pprint(self.config._fields)

            try:
                name = self.config.name
                self.log("Name set to: ", name, lvl=verbose)
            except (AttributeError, KeyError):  # pragma: no cover
                self.log("Has no name.", lvl=verbose)

            try:
                self.config.name = self.uniquename
            except (AttributeError, KeyError) as e:  # pragma: no cover
                self.log("Cannot set component name for configuration: ", e,
                         type(e), self.name, exc=True, lvl=critical)

            try:
                uuid = self.config.uuid
                self.log("UUID set to: ", uuid, lvl=verbose)
            except (AttributeError, KeyError):
                self.log("Has no UUID", lvl=verbose)
                self.config.uuid = str(uuid4())

            try:
                notes = self.config.notes
                self.log("Notes set to: ", notes, lvl=verbose)
            except (AttributeError, KeyError):
                self.log("Has no notes, trying docstring", lvl=verbose)

                notes = self.__doc__
                if notes is None:
                    notes = "No notes."
                else:
                    notes = notes.lstrip().rstrip()
                    self.log(notes)
                self.config.notes = notes

            try:
                componentclass = self.config.componentclass
                self.log("Componentclass set to: ", componentclass,
                         lvl=verbose)
            except (AttributeError, KeyError):
                self.log("Has no component class", lvl=verbose)
                self.config.componentclass = self.name

        except ValidationError as e:
            self.log("Not setting invalid component configuration: ", e,
                     type(e), exc=True, lvl=error)