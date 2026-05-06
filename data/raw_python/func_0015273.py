def default_icon_path(self):
        """Returns default path to icon of this assistant.

        Assuming self.path == "/foo/assistants/crt/python/django.yaml"
        For image format in [png, svg]:
            1) Take the path of this assistant and strip it of load path
               (=> "crt/python/django.yaml")
            2) Substitute its extension for <image format>
               (=> "crt/python/django.<image format>")
            3) Prepend self.load_path + 'icons'
               (=> "/foo/icons/crt/python/django.<image format>")
            4) If file from 3) exists, return it
        Return empty string if no icon found.
        """
        supported_exts = ['.png', '.svg']
        stripped = self.path.replace(os.path.join(self.load_path, 'assistants'), '').strip(os.sep)
        for ext in supported_exts:
            icon_with_ext = os.path.splitext(stripped)[0] + ext
            icon_fullpath = os.path.join(self.load_path, 'icons', icon_with_ext)
            if os.path.exists(icon_fullpath):
                return icon_fullpath
        return ''