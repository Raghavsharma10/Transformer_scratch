def Box(*args, **kw):
    """Axis-aligned box domain (same as AABox for now)

    WARNING: Deprecated, use AABox instead. This domain will mean something
    different in future versions of lepton
    """
    import warnings
    warnings.warn("lepton.domain.Box is deprecated, use AABox instead. "
                  "This domain class will mean something different in future versions of lepton",
                  stacklevel=2)
    return AABox(*args, **kw)