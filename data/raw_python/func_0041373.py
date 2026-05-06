def condition_from_text(text) -> Condition:
        """
        Return a Condition instance with PEG grammar from text

        :param text: PEG parsable string
        :return:
        """
        try:
            condition = pypeg2.parse(text, output.Condition)
        except SyntaxError:
            # Invalid conditions are possible, see https://github.com/duniter/duniter/issues/1156
            # In such a case, they are store as empty PEG grammar object and considered unlockable
            condition = Condition(text)
        return condition