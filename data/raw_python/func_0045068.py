def main():
    """
    main
    """
    args = MainArguments()

    if args.tool.lower() == "tool1":
        args = Tool1Arguments()
    elif args.tool.lower() == "tool2":
        args = Tool2Arguments()
    else:
        print("Unknown tool", args.tool)

    print(args)