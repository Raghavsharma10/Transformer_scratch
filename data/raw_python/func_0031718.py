def close(self):
        """
        Closes pipe
        :return:
        """
        resource = ResourceLocator(CommandShell.ShellResource)
        resource.add_selector('ShellId', self.__shell_id)
        self.session.delete(resource)