def may_be_null_is_nullable():
    """If may_be_null returns nullable or if NULL can be passed in.

    This can still be wrong if the specific typelib is older than the linked
    libgirepository.

    https://bugzilla.gnome.org/show_bug.cgi?id=660879#c47
    """

    repo = GIRepository()
    repo.require("GLib", "2.0", 0)
    info = repo.find_by_name("GLib", "spawn_sync")
    # this argument is (allow-none) and can never be (nullable)
    return not info.get_arg(8).may_be_null