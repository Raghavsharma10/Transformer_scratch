def quick_confirm(prompt, default_value):
        """
        Function to display a quick confirmation for user input

        **Parameters:**

          - **prompt:** Text to display before confirm
          - **default_value:** Default value for no entry

        **Returns:** 'y', 'n', or Default value.
        """
        valid = False
        value = default_value.lower()
        while not valid:
            input_val = compat_input(prompt + "[{0}]: ".format(default_value))

            if input_val == "":
                value = default_value.lower()
                valid = True
            else:
                try:
                    if input_val.lower() in ['y', 'n']:
                        value = input_val.lower()
                        valid = True
                    else:
                        print("ERROR: enter 'Y' or 'N'.")
                        valid = False

                except ValueError:
                    print("ERROR: enter 'Y' or 'N'.")
                    valid = False

        return value