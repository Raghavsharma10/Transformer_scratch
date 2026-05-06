def merge_tree(self, dst, symlinks=False, *args, **kwargs):
        """
        Copy entire contents of self to dst, overwriting existing
        contents in dst with those in self.

        If the additional keyword `update` is True, each
        `src` will only be copied if `dst` does not exist,
        or `src` is newer than `dst`.

        Note that the technique employed stages the files in a temporary
        directory first, so this function is not suitable for merging
        trees with large files, especially if the temporary directory
        is not capable of storing a copy of the entire source tree.
        """
        update = kwargs.pop('update', False)
        with tempdir() as _temp_dir:
            # first copy the tree to a stage directory to support
            #  the parameters and behavior of copytree.
            stage = _temp_dir / str(hash(self))
            self.copytree(stage, symlinks, *args, **kwargs)
            # now copy everything from the stage directory using
            #  the semantics of dir_util.copy_tree
            dir_util.copy_tree(stage, dst, preserve_symlinks=symlinks,
                update=update)