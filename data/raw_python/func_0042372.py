def get_profile(A):
    "Fail-soft profile getter; if no profile is present assume none and quietly ignore."
    try:
        with open(os.path.expanduser(A.profile)) as I:
            profile = json.load(I)
        return profile
    except:
        return {}