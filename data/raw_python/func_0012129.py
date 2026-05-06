def resolve_dst(self, dst_dir, src):
        """
        finds the destination based on source
        if source is an absolute path, and there's no pattern, it copies the file to base dst_dir
        """
        if os.path.isabs(src):
            return os.path.join(dst_dir, os.path.basename(src))
        return os.path.join(dst_dir, src)