def splitext(p):
    r"""Like the normal splitext (in posixpath), but doesn't treat dotfiles
    (e.g. .emacs) as extensions. Also uses os.sep instead of '/'."""

    root, ext = os.path.splitext(p)
    # check for dotfiles
    if (not root or root[-1] == os.sep): # XXX: use '/' or os.sep here???
        return (root + ext, "")
    else:
        return root, ext