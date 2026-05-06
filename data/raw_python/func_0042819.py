def get_alias(self, alias):
        """
        RETURN REFERENCE TO ALIAS (MANY INDEXES)
        USER MUST BE SURE NOT TO SEND UPDATES
        """
        aliases = self.get_aliases()
        if alias in aliases.alias:
            settings = self.settings.copy()
            settings.alias = alias
            settings.index = alias
            return Index(read_only=True, kwargs=settings, cluster=self)
        Log.error("Can not find any index with alias {{alias_name}}",  alias_name= alias)