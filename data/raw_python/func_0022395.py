def quick_int_input(prompt, default_value, min_val=1, max_val=30):
        """
        Function to display a quick question for integer user input

        **Parameters:**

          - **prompt:** Text / question to display
          - **default_value:** Default value for no entry
          - **min_val:** Lowest allowed integer
          - **max_val:** Highest allowed integer

        **Returns:** integer or default_value.
        """
        valid = False
        num_val = default_value
        while not valid:
            input_val = compat_input(prompt + "[{0}]: ".format(default_value))

            if input_val == "":
                num_val = default_value
                valid = True
            else:
                try:
                    num_val = int(input_val)
                    if min_val <= num_val <= max_val:
                        valid = True
                    else:
                        print("ERROR: must be between {0} and {1}.".format(min, max))
                        valid = False

                except ValueError:
                    print("ERROR: must be a number.")
                    valid = False

        return num_val