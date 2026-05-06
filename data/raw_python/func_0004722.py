def ast_for_stmts(self, stmts: T) -> None:
        """
        Stmts    ::= TokenDef{0, 1} Equals*;
        """
        if not stmts:
            raise ValueError('no ast found!')
        head, *equals = stmts

        if head.name is NameEnum.TokenDef:
            self.ast_for_token_def(head)
        elif head.name is NameEnum.TokenIgnore:
            self.ast_for_token_ignore(head)
        else:
            self.ast_for_equals(head)

        for each in equals:
            self.ast_for_equals(each)

        # if every combined parser can reach any other combined, 
        # just take any of them and compile it!
        if not self.compile_helper.alone and self._current__combined_parser_name:
            self.compile_helper.alone.add(self._current__combined_parser_name)