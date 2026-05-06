def create_result(self, local_path, container_path, permissions, meta, val, dividers):
        """Default permissions to rw"""
        if permissions is NotSpecified:
            permissions = 'rw'
        return Mount(local_path, container_path, permissions)