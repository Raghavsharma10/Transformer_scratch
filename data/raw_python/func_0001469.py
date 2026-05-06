def _collapse_invariants(bases: List[type], namespace: MutableMapping[str, Any]) -> None:
    """Collect invariants from the bases and merge them with the invariants in the namespace."""
    invariants = []  # type: List[Contract]

    # Add invariants of the bases
    for base in bases:
        if hasattr(base, "__invariants__"):
            invariants.extend(getattr(base, "__invariants__"))

    # Add invariants in the current namespace
    if '__invariants__' in namespace:
        invariants.extend(namespace['__invariants__'])

    # Change the final invariants in the namespace
    if invariants:
        namespace["__invariants__"] = invariants