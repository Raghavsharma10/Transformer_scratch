def check_arrange_act_spacing(self) -> typing.Generator[AAAError, None, None]:
        """
        * When no spaces found, point error at line above act block
        * When too many spaces found, point error at 2nd blank line
        """
        yield from self.check_block_spacing(
            LineType.arrange,
            LineType.act,
            'AAA03 expected 1 blank line before Act block, found {}',
        )