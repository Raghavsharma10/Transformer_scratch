def get_permissions(self):
        """
        Soyut role ait Permission nesnelerini bulur ve code değerlerini
        döner.

        Returns:
            list: Permission code değerleri

        """
        return [p.permission.code for p in self.Permissions if p.permission.code]