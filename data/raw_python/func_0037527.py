def menu(items, heading):
    '''Takes list of dictionaries and prints a menu.
        items parameter should be in the form of a list, containing
        dictionaries with the keys: {"key", "text", "function"}.

        Typing the key for a menuitem, followed by return, will run
        "function".
    '''

    heading = "\n"*5 + heading      # A little vertical padding

    while True:
        keydict = {}

        clear_screen()
        print(heading)

        for item in items:
            menustring = "  " + item["key"] + " " + item["text"]
            keydict[item["key"]] = item["function"]
            print(menustring)

        key = input("\nType key and Return (q to quit): ").strip()

        if key.lower() == "q":
            return
        else:
            try:
                ret = keydict[key]()
                if ret:    # If child returns non-false, exit menu.
                    return 1
            except KeyError: # Handle garbage input.
                continue