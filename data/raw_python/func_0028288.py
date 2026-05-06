def run(self):
        """
        Run command.

        :return: Command exit code.
        """
        # Print task title
        print_title(self._task_name_title)

        # Get context object
        ctx = self._ctx

        # Get command parts
        cmd_part_s = self._parts

        # Print title
        print_title('Find program')

        # Get program path from command parts
        program_path = cmd_part_s[0]

        # If the program path ends with one of the file extensions below
        if program_path.endswith(('.exe', '.com', '.bat', '.cmd')):
            # Remove the file extension.
            # This is because `Waf`'s `find_binary` function adds each of the
            # file extensions when finding the program.
            program_path = program_path[:-4]

        # Print info
        print_text('Find program: {0}'.format(program_path))

        #
        try:
            # Find program
            found_program_path_s = ctx.find_program(program_path, quiet=True)

        # If program paths are not found
        except ConfigurationError:
            # Get error message
            msg = 'Error (2D7VS): Program is not found: {0}'.format(
                program_path
            )

            # Raise error
            raise ValueError(msg)

        # If program paths are found.

        # If program paths are found.
        #     Use the first program path found.
        # If program paths are not found. (Should not happen.)
        #     Use given program path.
        found_program_path = found_program_path_s[0] \
            if found_program_path_s else program_path

        # Use the program path found as the first command part
        cmd_part_s[0] = found_program_path

        # Print info
        print_text('Use program: {0}'.format(found_program_path))

        # Print end title
        print_title('Find program', is_end=True)

        # Print title
        print_title('PATH')

        # Print environment variable PATH's value, one part per line
        print_text('\n'.join(os.environ.get('PATH', '').split(os.pathsep)))

        # Print end title
        print_title('PATH', is_end=True)

        # Print title
        print_title('PYTHONPATH')

        # Print environment variable PYTHONPATH's value, one part per line
        print_text(
            '\n'.join(os.environ.get('PYTHONPATH', '').split(os.pathsep))
        )

        # Print end title
        print_title('PYTHONPATH', is_end=True)

        # Print title
        print_title('DIR')

        # Print working directory
        print_text(self._cwd)

        # Print end title
        print_title('DIR', is_end=True)

        # Print title
        print_title('CMD')

        # Print the command in multi-line format
        print_text(_format_multi_line_command(cmd_part_s))

        # Print end title
        print_title('CMD', is_end=True)

        # Print title
        print_title('RUN')

        # Run the command in the working directory
        exit_code = self.exec_command(cmd_part_s, cwd=self._cwd)

        # Print the command's exit code
        print_text('Exit code: {0}'.format(exit_code))

        # Print end title
        print_title('RUN', is_end=True)

        # Print task end title
        print_title(self._task_name_title, True)

        # Return the exit code
        return exit_code