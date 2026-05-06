def contains_repository(cls, context, directory):
        """
        Check whether the given directory contains a local repository.

        :param directory: The pathname of a directory (a string).
        :returns: :data:`True` if the directory contains a local repository,
                  :data:`False` otherwise.

        By default :func:`contains_repository()` just checks whether the
        directory reported by :func:`get_vcs_directory()` exists, but
        :class:`Repository` subclasses can override this class method to
        improve detection accuracy.
        """
        return context.is_directory(cls.get_vcs_directory(context, directory))