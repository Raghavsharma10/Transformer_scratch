def versioning_models_registered(manager, base):
    """Return True if all versioning models have been registered."""
    declared_models = base._decl_class_registry.keys()
    return all(versioning_model_classname(manager, c) in declared_models
               for c in manager.pending_classes)