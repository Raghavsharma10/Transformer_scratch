def check_act_assert_spacing(self) -> typing.Generator[AAAError, None, None]:
        """
        * When no spaces found, point error at line above assert block
        * When too many spaces found, point error at 2nd blank line
        """
        yield from self.check_block_spacing(
            LineType.act,
            LineType._assert,
            'AAA04 expected 1 blank line before Assert block, found {}',
        )