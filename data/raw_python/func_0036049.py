def get_settings(all,key):
    """View Hitman internal settings. Use 'all' for all keys"""
    with Database("settings") as s:
        if all:
            for k, v in zip(list(s.keys()), list(s.values())):
                print("{} = {}".format(k, v))
        elif key:
            print("{} = {}".format(key, s[key]))
        else:
            print("Don't know what you want? Try --all")