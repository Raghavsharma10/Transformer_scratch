def _pfp__set_value(self, new_val):
        """Set the value of the String, taking into account
        escaping and such as well
        """
        if not isinstance(new_val, Field):
            new_val = utils.binary(utils.string_escape(new_val))
        super(String, self)._pfp__set_value(new_val)