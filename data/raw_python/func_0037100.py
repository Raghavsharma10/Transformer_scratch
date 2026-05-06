def buildCommands(self,files,args):
        """
        Given a list of (input) files, buildCommands builds all the commands.
        This is one of the two key methods of MapExecutor.
        """
        commands = []
        count = args.count_from
        # For each file, a command is created:
        for fileName in files:
            commands.append(self.buildCommand(fileName,count,args))
            count = count+1
        return commands