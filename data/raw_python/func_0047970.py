def find_files(self, context, silent_build):
        """
        Find the set of files from our parent_dir that we care about
        """
        first_layer = ["'{0}'".format(thing) for thing in os.listdir(context.parent_dir)]
        output, status = command_output("find {0} -type l -or -type f {1} -follow -print".format(' '.join(first_layer), context.find_options), cwd=context.parent_dir)
        if status != 0:
            if context.ignore_find_errors:
                log.warning("The find command failed to run, will continue anyway")
            else:
                raise HarpoonError("Couldn't find the files we care about", output=output, cwd=context.parent_dir)
        all_files = set(self.convert_nonascii(output))
        total_files = set(all_files)

        combined = set(all_files)

        if context.use_gitignore:
            if context.parent_dir == context.git_root:
                all_files = set([path for path in all_files if not path.startswith(".git")])

            combined = set(all_files)
            valid_files = self.find_notignored_git_files(context, silent_build)

            removed = set()
            if valid_files:
                for fle in combined:
                    if fle not in valid_files:
                        removed.add(fle)
            if removed and not silent_build: log.info("Ignoring %s/%s files", len(removed), len(combined))
            combined -= removed

        if context.exclude:
            excluded = set()
            for filename in combined:
                for excluder in context.exclude:
                    if fnmatch.fnmatch(filename, excluder):
                        excluded.add(filename)
                        break
            if not silent_build: log.info("Filtering %s/%s items\texcluding=%s", len(excluded), len(combined), context.exclude)
            combined -= excluded

        if context.include:
            extra_included = []
            for filename in total_files:
                for includer in context.include:
                    if fnmatch.fnmatch(filename, includer):
                        extra_included.append(filename)
                        break
            if not silent_build: log.info("Adding back %s items\tincluding=%s", len(extra_included), context.include)
            combined = set(list(combined) + extra_included)

        files = sorted(os.path.join(context.parent_dir, filename) for filename in combined)
        if not silent_build: log.info("Adding %s things from %s to the context", len(files), context.parent_dir)
        return files