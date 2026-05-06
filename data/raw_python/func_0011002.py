def get_files(commit_only=True, copy_dest=None):
    "Get copies of files for analysis."
    if commit_only:
        real_files = bash(
            "git diff --cached --name-status | "
            "grep -v -E '^D' | "
            "awk '{ print ( $(NF) ) }' "
        ).value().strip()
    else:
        real_files = bash(
            "git ls-tree --name-only --full-tree -r HEAD"
        ).value().strip()

    if real_files:
        return create_fake_copies(real_files.split('\n'), copy_dest)
    return []