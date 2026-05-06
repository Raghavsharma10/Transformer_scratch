def get(self, field, value=None):
        """Gets user input for given field and checks if it is valid.

        If input is invalid, it will ask the user to enter it again.
        Defaults values to empty or :value:.

        It does not check validity of parent index. It can only be tested
        further down the road, so for now accept anything.

        :field: Field name.
        :value: Default value to use for field.
        :returns: User input.

        """
        self.value = value
        val = self.input(field)
        if field == 'name':
            while True:
                if val != '':
                    break
                print("Name cannot be empty.")
                val = self.input(field)
        elif field == 'priority':
            if val == '':  # Use default priority
                return None
            while True:
                if val in Get.PRIORITIES.values():
                    break
                c, val = val, Get.PRIORITIES.get(val)
                if val:
                    break
                print("Unrecognized priority number or name [{}].".format(c))
                val = self.input(field)
            val = int(val)
        return val