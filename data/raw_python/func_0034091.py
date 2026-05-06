def input(self, field):
        """Gets user input for given field.

        Can be interrupted with ^C.

        :field: Field name.
        :returns: User input.

        """
        try:
            desc = Get.TYPES[field]
            return input("{}|{}[{}]> ".format(
                field, "-" * (Get._LEN - len(field) - len(desc)), desc
            ))
        except KeyboardInterrupt:
            print()
            exit(0)