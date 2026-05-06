def quick_str_input(prompt, default_value):
        """
        Function to display a quick question for text input.

        **Parameters:**

          - **prompt:** Text / question to display
          - **default_value:** Default value for no entry

        **Returns:** text_type() or default_value.
        """
        valid = False
        str_val = default_value
        while not valid:
            input_val = raw_input(prompt + "[{0}]: ".format(default_value))

            if input_val == "":
                str_val = default_value
                valid = True
            else:
                try:
                    str_val = text_type(input_val)
                    valid = True

                except ValueError:
                    print("ERROR: must be text.")
                    valid = False

        return str_val