def progress(self):
        """
            Deal with next token
            Used to create, fillout and end groups and singles
            As well as just append everything else
        """
        tokenum, value, scol = self.current.values()

        # Default to not appending anything
        just_append = False

        # Prevent from group having automatic pass given to it
        # If it already has a pass
        if tokenum == NAME and value == 'pass':
            self.groups.empty = False

        # Set variables to be used later on to determine if this will likely make group not empty
        created_group = False
        found_content = False
        if not self.groups.starting_group and not self.is_space:
            found_content = True

        if self.groups.starting_group:
            # Inside a group signature, add to it
            if tokenum == STRING:
                self.groups.name = value

            elif tokenum == NAME or (tokenum == OP and value == '.'):
                # Modify super class for group
                self.groups.modify_kls(value)

            elif tokenum == NEWLINE:
                # Premature end of group
                self.add_tokens_for_group(with_pass=True)

            elif tokenum == OP and value == ":":
                # Proper end of group
                self.add_tokens_for_group()

        elif self.groups.starting_single:
            # Inside single signature, add to it
            if tokenum == STRING:
                self.single.name = value

            elif tokenum == NEWLINE and not self.in_container:
                # Premature end of single
                self.add_tokens_for_single(ignore=True)

            elif tokenum == OP and value == ":":
                # Proper end of single
                self.add_tokens_for_single()

            elif value and self.single.name:
                # Only want to add args after the name for the single has been specified
                self.single.add_to_arg(tokenum, value)

        elif self.after_space or self.after_an_async or scol == 0 and tokenum == NAME:
            # set after_an_async if we found an async by itself
            # So that we can just have that prepended and still be able to interpret our special blocks
            with_async = self.after_an_async
            if not self.after_an_async and value == "async":
                self.after_an_async = True
            else:
                self.after_an_async = False

            if value in ('describe', 'context'):
                created_group = True

                # add pass to previous group if nothing added between then and now
                if self.groups.empty and not self.groups.root:
                    self.add_tokens_for_pass()

                # Start new group
                self.groups = self.groups.start_group(scol, value)
                self.all_groups.append(self.groups)

            elif value in ('it', 'ignore'):
                self.single = self.groups.start_single(value, scol)

            elif value in ('before_each', 'after_each'):
                setattr(self.groups, "has_%s" % value, True)
                if with_async:
                    setattr(self.groups, "async_%s" % value, True)
                self.add_tokens_for_test_helpers(value, with_async=with_async)

            else:
                just_append = True
        else:
            # Don't care about it, append!
            just_append = True

        # Found something that isn't whitespace or a new group
        # Hence current group isn't empty !
        if found_content and not created_group:
            self.groups.empty = False

        # Just append if token should be
        if just_append:
            self.result.append([tokenum, value])