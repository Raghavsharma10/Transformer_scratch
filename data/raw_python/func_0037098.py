def replaceInCommand(self,command, pattern, replacement, replacementAtBeginning):
        """
        This is in internal method that replaces a certain 'pattern' in the
        provided command with a 'replacement'.
        A different replacement can be specified when the pattern occurs right
        at the beginning of the command.
        """
        # Turn the command into a list:
        commandAsList = list(command)
        # Get the indices of the pattern in the list:
        indices = [index.start() for index in re.finditer(pattern, command)]
        # Replace at the indices, unless the preceding character is the
        # escape character:
        for index in indices:
            if index == 0:
                commandAsList[index] = replacementAtBeginning
            elif commandAsList[index-1] != MapConstants.escape_char:
                commandAsList[index] = replacement
        # Put the pieces of the new command together:
        newCommand = ''.join(commandAsList)
        # Remove superfluous slashes and return:
        return newCommand.replace("//","/")