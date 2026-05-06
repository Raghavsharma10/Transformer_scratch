def compute_numtab(self):
        """
        Recomputes the sets for the static ranges of the trigger time.

        This method should only be called by the user if the string_tab
        member is modified.
        """
        self.numerical_tab = []

        for field_str, span in zip(self.string_tab, FIELD_RANGES):
            split_field_str = field_str.split(',')
            if len(split_field_str) > 1 and "*" in split_field_str:
                raise ValueError("\"*\" must be alone in a field.")

            unified = set()
            for cron_atom in split_field_str:
                # parse_atom only handles static cases
                if not(is_special_atom(cron_atom, span)):
                    unified.update(parse_atom(cron_atom, span))

            self.numerical_tab.append(unified)

        if self.string_tab[2] == "*" and self.string_tab[4] != "*":
            self.numerical_tab[2] = set()
        elif self.string_tab[4] == "*" and self.string_tab[2] != "*":
            self.numerical_tab[4] = set()