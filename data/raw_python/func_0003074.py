def get(self,style):
        """ what's the value of a style at the current stack level"""
        level = len(self.stack) -1
        while level >= 0:
            if style in self.stack[level]:
                return self.stack[level][style]
            else:
                level = level - 1
        return None