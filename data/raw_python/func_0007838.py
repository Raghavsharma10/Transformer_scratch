def __sepApp(self, IDs, aspList):
        """ Returns true if the object last and next movement are
        separations and applications to objects in list IDs.
        It only considers aspects in aspList.
        
        This function is static since it does not test if the next
        application will be indeed perfected. It considers only
        a snapshot of the chart and not its astronomical movement.
        
        """
        sep, app = self.dyn.immediateAspects(self.obj.id, aspList)
        if sep is None or app is None:
            return False
        else:
            sepCondition = sep['id'] in IDs
            appCondition = app['id'] in IDs
            return sepCondition == appCondition == True