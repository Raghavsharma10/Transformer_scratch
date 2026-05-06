def check(self, gen):
        """Check that the last part of the chain matches.

        TODO: Fix to handle the following situation that appears to not work

        say 'message 1'
        play sound until done
        say 'message 2'
        say 'message 3'
        play sound until done
        say ''

        """
        retval = Counter()
        name, _, block = next(gen, ('', 0, ''))
        if name in self.SAY_THINK:
            if self.is_blank(block.args[0]):
                retval[self.CORRECT] += 1
            else:
                name, _, block = next(gen, ('', 0, ''))
                if name == 'play sound %s until done':
                    # Increment the correct count because we have at least
                    # one successful instance
                    retval[self.CORRECT] += 1
                    # This block represents the beginning of a second
                    retval += self.check(gen)
                else:
                    retval[self.INCORRECT] += 1
        else:
            retval[self.INCORRECT] += 1
        return retval