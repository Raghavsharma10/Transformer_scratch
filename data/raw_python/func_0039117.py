def _get_branches(self):
        """Get branches from org/repo."""
        if self.offline:
            local_path = Path(LOCAL_PATH).expanduser() / self.org / self.repo
            get_refs = f"git -C {shlex.quote(str(local_path))} show-ref --heads"
        else:
            get_refs = f"git ls-remote --heads https://github.com/{self.org}/{self.repo}"
        try:
            # Parse get_refs output for the actual branch names
            return (line.split()[1].replace("refs/heads/", "") for line in _run(get_refs, timeout=3).split("\n"))
        except Error:
            return []