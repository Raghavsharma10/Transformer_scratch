def ignore_token(self):
        """Determine if we should ignore current token"""
        def get_next_ignore(remove=False):
            """Get next ignore from ignore_next and remove from ignore_next"""
            next_ignore = self.ignore_next

            # Just want to return it, don't want to remove yet
            if not remove:
                if type(self.ignore_next) in (list, tuple):
                    next_ignore = self.ignore_next[0]
                return next_ignore

            # Want to remove it from ignore_next
            if type(next_ignore) in (list, tuple) and next_ignore:
                next_ignore = self.ignore_next.pop(0)
            elif not next_ignore:
                self.next_ignore = None
                next_ignore = None
            else:
                self.next_ignore = None

            return next_ignore

        # If we have tokens to be ignored and we're not just inserting till some token
        if not self.insert_till and self.ignore_next:
            # Determine what the next ignore is
            next_ignore = get_next_ignore()
            if next_ignore == (self.current.tokenum, self.current.value):
                # Found the next ignore token, remove it from the stack
                # So that the next ignorable token can be considered
                get_next_ignore(remove=True)
                return True
            else:
                # If not a wildcard, then return now
                if type(next_ignore) is not WildCard:
                    return False

                # Go through tokens untill we find one that isn't a wildcard
                while type(next_ignore) == WildCard:
                    next_ignore = get_next_ignore(remove=True)

                # If the next token is next ignore then we're done here!
                if next_ignore == (self.current.tokenum, self.current.value):
                    get_next_ignore(remove=True)
                    return True
                else:
                    # If there is another token to ignore, then consider the wildcard
                    # And keep inserting till we reach this next ignorable token
                    if next_ignore:
                        self.insert_till = next_ignore
                    return False