def get_path_signature(self, path):
        """generate a unique signature for file contained in path
        """
        if not os.path.exists(path):
            return None
        if os.path.isdir(path):
            merge = {}
            for root, dirs, files in os.walk(path):
                for name in files:
                    full_name = os.path.join(root, name)
                    merge[full_name] = os.stat(full_name)
            return merge
        else:
            return os.stat(path)