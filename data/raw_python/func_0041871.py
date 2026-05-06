def _get_corresponding_parsers(self, func):
        """Get the parser that has been set up by the given `function`"""
        if func in self._used_functions:
            yield self
        if self._subparsers_action is not None:
            for parser in self._subparsers_action.choices.values():
                for sp in parser._get_corresponding_parsers(func):
                    yield sp