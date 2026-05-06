def find_notignored_git_files(self, context, silent_build):
        """
        Return a list of files that are not ignored by git
        """
        def git(args, error_message, cwd=context.parent_dir, **error_kwargs):
            output, status = command_output("git {0}".format(args), cwd=cwd)
            if status != 0:
                error_kwargs['output'] = output
                error_kwargs['directory'] = context.parent_dir
                raise HarpoonError(error_message, **error_kwargs)
            return output

        changed_files = git("diff --name-only", "Failed to determine what files have changed")
        untracked_files = git("ls-files --others --exclude-standard", "Failed to find untracked files")

        valid = set()
        under_source_control = git("ls-files --exclude-standard", "Failed to find all the files under source control")
        git_submodules = [regexes["whitespace"].split(line.strip())[1] for line in git("submodule status", "Failed to find submodules", cwd=context.git_root)]
        git_submodules = [os.path.normpath(os.path.relpath(os.path.abspath(p), os.path.abspath(os.path.relpath(context.parent_dir, context.git_root)))) for p in git_submodules]

        valid = under_source_control + untracked_files

        for filename in list(valid):
            matched = False
            if context.exclude:
                for excluder in context.exclude:
                    if fnmatch.fnmatch(filename, excluder):
                        matched = True
                        break

            if matched:
                continue

            location = os.path.join(context.parent_dir, filename)
            if os.path.islink(location) and os.path.isdir(location):
                actual_path = os.path.abspath(os.path.realpath(location))
                parent_dir = os.path.abspath(os.path.realpath(context.parent_dir))
                include_from = os.path.relpath(actual_path, parent_dir)

                to_include = git("ls-files --exclude-standard -- {0}".format(include_from), "Failed to find files under a symlink")
                for found in to_include:
                    valid += [os.path.join(filename, os.path.relpath(found, include_from))]
            elif os.path.isdir(location) and filename in git_submodules:
                to_include = git("ls-files --exclude-standard", "Failed to find files in a submodule", cwd=location)
                valid = [v for v in valid if v != filename]
                for found in to_include:
                    valid.append(os.path.join(filename, found))

        return set(self.convert_nonascii(valid))