def preferences_class_prepared(sender, *args, **kwargs):
    """
    Adds various preferences members to preferences.preferences,
    thus enabling easy access from code.
    """
    cls = sender
    if issubclass(cls, Preferences):
        # Add singleton manager to subclasses.
        cls.add_to_class('singleton', SingletonManager())
        # Add property for preferences object to preferences.preferences.
        setattr(preferences.Preferences, cls._meta.object_name, property(lambda x: cls.singleton.get()))