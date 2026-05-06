def main():
    """Main entry-point for oz's cli"""

    # Hack to make user code available for import
    sys.path.append(".")

    # Run the specified action
    oz.initialize()
    retr = optfn.run(list(oz._actions.values()))

    if retr == optfn.ERROR_RETURN_CODE:
        sys.exit(-1)
    elif retr == None:
        sys.exit(0)
    elif isinstance(retr, int):
        sys.exit(retr)
    else:
        raise Exception("Unexpected return value from action: %s" % retr)