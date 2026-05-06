def nextClass(self, classuri):
        """Returns the next class in the list of classes. If it's the last one, returns the first one."""
        if classuri == self.classes[-1].uri:
            return self.classes[0]
        flag = False
        for x in self.classes:
            if flag == True:
                return x
            if x.uri == classuri:
                flag = True
        return None