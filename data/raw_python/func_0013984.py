def add_class(self, ioclass):
        """Add one VNXIOClass instance to policy.

        .. note: due to the limitation of VNX, need to stop the policy first.
        """
        current_ioclasses = self.ioclasses
        if ioclass.name in current_ioclasses.name:
            return
        current_ioclasses.append(ioclass)
        self.modify(new_ioclasses=current_ioclasses)