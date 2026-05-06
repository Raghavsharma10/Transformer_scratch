def _installV2Powerups(anonymousSite):
    """
    Install the given L{AnonymousSite} for the powerup interfaces it was given
    in version 2.
    """
    anonymousSite.store.powerUp(anonymousSite, IWebViewer)
    anonymousSite.store.powerUp(anonymousSite, IMantissaSite)