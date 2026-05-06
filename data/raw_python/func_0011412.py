def _pfp__set_value(self, value):
        """Initialize the struct. Value should be an array of
        fields, one each for each struct member.

        :value: An array of fields to initialize the struct with
        :returns: None
        """
        if self._pfp__frozen:
            raise errors.UnmodifiableConst()
        if len(value) != len(self._pfp__children):
            raise errors.PfpError("struct initialization has wrong number of members")

        for x in six.moves.range(len(self._pfp__children)):
            self._pfp__children[x]._pfp__set_value(value[x])