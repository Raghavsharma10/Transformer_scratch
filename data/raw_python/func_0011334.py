def del_role(self, role):
        """ deletes a group """
        target = AuthGroup.objects(role=role, creator=self.client).first()
        if target:
            target.delete()
            return True
        else:
            return False