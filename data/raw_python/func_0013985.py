def remove_class(self, ioclass):
        """Remove VNXIOClass instance from policy."""
        current_ioclasses = self.ioclasses
        new_ioclasses = filter(lambda x: x.name != ioclass.name,
                               current_ioclasses)
        self.modify(new_ioclasses=new_ioclasses)