def quick_menu(self, banner, list_line_format, choice_list):
        """
        Function to display a quick menu for user input

        **Parameters:**

          - **banner:** Text to display before menu
          - **list_line_format:** Print'ing string with format spots for index + tuple values
          - **choice_list:** List of tuple values that you want returned if selected (and printed)

        **Returns:** Tuple that was selected.
        """
        # Setup menu
        invalid = True
        menu_int = -1

        # loop until valid
        while invalid:
            print(banner)

            for item_index, item_value in enumerate(choice_list):
                print(list_line_format.format(item_index + 1, *item_value))

            menu_choice = compat_input("\nChoose a Number or (Q)uit: ")

            if str(menu_choice).lower() in ['q']:
                # exit
                print("Exiting..")
                # best effort logout
                self._parent_class.get.logout()
                sys.exit(0)

            # verify number entered
            try:
                menu_int = int(menu_choice)
                sanity = True
            except ValueError:
                # not a number
                print("ERROR: ", menu_choice)
                sanity = False

            # validate number chosen
            if sanity and 1 <= menu_int <= len(choice_list):
                invalid = False
            else:
                print("Invalid input, needs to be between 1 and {0}.\n".format(len(choice_list)))

        # return the choice_list tuple that matches the entry.
        return choice_list[int(menu_int) - 1]