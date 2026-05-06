def main(self, auto=None, loop=False, quit=("q", "Quit"), **kwargs):
        """Runs the standard menu main logic. Any `kwargs` supplied will be
        pass to `Menu.show()`. If `argv` is provided to the script, it will be
        used as the `auto` parameter.

        **Params**:
          - auto ([str]) - If provided, the list of strings with be used as
            input for the menu prompts.
          - loop (bool) - If true, the menu will loop until quit.
          - quit ((str,str)) - If provided, adds a quit option to the menu.
        """
        def _main():
            global _AUTO
            if quit:
                if self.entries[-1][:2] != quit:
                    self.add(*quit, func=lambda: quit[0])
            if stdin_auto.auto:
                _AUTO = True
            result = None
            if loop:
                note = "Menu loops until quit."
                try:
                    while True:
                        mresult = self.show(note=note, **kwargs)
                        if mresult in quit:
                            break
                        result = mresult
                except EOFError:
                    pass
                return result
            else:
                note = "Menu does not loop, single entry."
                result = self.show(note=note, **kwargs)
            return result
        global _AUTO
        if _AUTO:
            return _main()
        else:
            with stdin_auto:
                return _main()