def get_resizer(self, size, target_size):
        '''Choose a resizer depending an image size'''
        sw, sh = size
        if sw >= sh * self.rate:
            return self.hor_resize
        else:
            return self.vert_resize