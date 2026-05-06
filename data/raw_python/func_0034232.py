def configure_specials_key(self, keyboard):
        """Configures specials key if needed.

        :param keyboard: Keyboard instance this layout belong.
        """
        special_row = VKeyRow()
        max_length = self.max_length
        i = len(self.rows) - 1
        current_row = self.rows[i]
        special_keys = [VBackKey()]
        if self.allow_uppercase: special_keys.append(VUppercaseKey(keyboard))
        if self.allow_special_chars: special_keys.append(VSpecialCharKey(keyboard))
        while len(special_keys) > 0:
            first = False
            while len(special_keys) > 0 and len(current_row) < max_length:
                current_row.add_key(special_keys.pop(0), first=first)
                first = not first
            if i > 0:
                i -= 1
                current_row = self.rows[i]
            else:
                break
        if self.allow_space:
            space_length = len(current_row) - len(special_keys)
            special_row.add_key(VSpaceKey(space_length))
        first = True
        # Adding left to the special bar.
        while len(special_keys) > 0:
            special_row.add_key(special_keys.pop(0), first=first)
            first = not first
        if len(special_row) > 0:
            self.rows.append(special_row)