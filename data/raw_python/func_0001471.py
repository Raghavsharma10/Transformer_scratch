def _collapse_snapshots(base_snapshots: List[Snapshot], snapshots: List[Snapshot]) -> List[Snapshot]:
    """
    Collapse snapshots of pre-invocation values with the snapshots collected from the base classes.

    :param base_snapshots: snapshots collected from the base classes
    :param snapshots: snapshots of the function (before the collapse)
    :return: collapsed sequence of snapshots
    """
    seen_names = set()  # type: Set[str]
    collapsed = base_snapshots + snapshots

    for snap in collapsed:
        if snap.name in seen_names:
            raise ValueError("There are conflicting snapshots with the name: {!r}.\n\n"
                             "Please mind that the snapshots are inherited from the base classes. "
                             "Does one of the base classes defines a snapshot with the same name?".format(snap.name))

        seen_names.add(snap.name)

    return collapsed