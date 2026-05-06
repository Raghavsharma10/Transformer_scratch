def display_results(repo_name, contributors, api_len):
    """
    Fancy display. 
    """
    print("\n")

    print("All Contributors:")

    # Sort and consolidate on Name
    seen = []
    for user in sorted(contributors, key=_sort_by_name):
        if user.get("name"):
            key = user["name"]
        else:
            key = user["user_name"]
        if key not in seen:
            seen.append(key)
            if key != user["user_name"]:
                print("%s (%s)" % (user["name"], user["user_name"]))
            else:
                print(user["user_name"])

    print("")

    print("Repo: %s" % repo_name)
    print("GitHub Contributors: %s" % api_len)
    print("All Contributors: %s 👏" % len(seen))