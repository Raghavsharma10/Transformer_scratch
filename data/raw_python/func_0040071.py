def prepare(self):
    """
    Prepares the android project to the build process.

    Checks if the project uses either gradle or ant and executes the necessary steps.
    """
    self.src_folder = self.get_src_folder()
    st = os.stat('%s/gradlew' % self.path)
    os.chmod('%s/gradlew' % self.path, st.st_mode | stat.S_IEXEC)