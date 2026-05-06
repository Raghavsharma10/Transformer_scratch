def current_branch(self):
        """The name of the branch that's currently checked out in the working tree (a string or :data:`None`)."""
        output = self.context.capture('git', 'rev-parse', '--abbrev-ref', 'HEAD', check=False, silent=True)
        return output if output != 'HEAD' else None