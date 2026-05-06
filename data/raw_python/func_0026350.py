def main():
    """Command line interface for the ``qpass`` program."""
    # Initialize logging to the terminal.
    coloredlogs.install()
    # Prepare for command line argument parsing.
    action = show_matching_entry
    program_opts = dict(exclude_list=[])
    show_opts = dict(filters=[], use_clipboard=is_clipboard_supported())
    verbosity = 0
    # Parse the command line arguments.
    try:
        options, arguments = getopt.gnu_getopt(
            sys.argv[1:],
            "elnp:f:x:vqh",
            ["edit", "list", "no-clipboard", "password-store=", "filter=", "exclude=", "verbose", "quiet", "help"],
        )
        for option, value in options:
            if option in ("-e", "--edit"):
                action = edit_matching_entry
            elif option in ("-l", "--list"):
                action = list_matching_entries
            elif option in ("-n", "--no-clipboard"):
                show_opts["use_clipboard"] = False
            elif option in ("-p", "--password-store"):
                stores = program_opts.setdefault("stores", [])
                stores.append(PasswordStore(directory=value))
            elif option in ("-f", "--filter"):
                show_opts["filters"].append(value)
            elif option in ("-x", "--exclude"):
                program_opts["exclude_list"].append(value)
            elif option in ("-v", "--verbose"):
                coloredlogs.increase_verbosity()
                verbosity += 1
            elif option in ("-q", "--quiet"):
                coloredlogs.decrease_verbosity()
                verbosity -= 1
            elif option in ("-h", "--help"):
                usage(__doc__)
                return
            else:
                raise Exception("Unhandled option! (programming error)")
        if not (arguments or action == list_matching_entries):
            usage(__doc__)
            return
    except Exception as e:
        warning("Error: %s", e)
        sys.exit(1)
    # Execute the requested action.
    try:
        show_opts["quiet"] = verbosity < 0
        kw = show_opts if action == show_matching_entry else {}
        action(QuickPass(**program_opts), arguments, **kw)
    except PasswordStoreError as e:
        # Known issues don't get a traceback.
        logger.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        # If the user interrupted an interactive prompt they most likely did so
        # intentionally, so there's no point in generating more output here.
        sys.exit(1)