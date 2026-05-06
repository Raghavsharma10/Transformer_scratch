def up_to(self, key):
        '''Gets the recently inserted values up to a key'''
        for okey, ovalue in reversed(self.history):
            if okey == key:
                break
            else:
                yield ovalue