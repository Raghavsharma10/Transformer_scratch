def validate(self):
    """
    Validates the app project before the build.

    This is the first step in the build process.

    Needs to be implemented by the subclass.
    """
    if os.path.exists('%s/gradlew' % self.path) is False:
      raise errors.InvalidProjectStructure(message='Missing gradlew project root folder')

    self.touch_log('validate')