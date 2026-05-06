def adjust_size(self, dst_x, dst_y, mode=FIT):
        """
        given a x and y of dest, determine the ratio and return
        an (x,y,w,h) for a output image.
        """
        # get image size
        image = Image.open(self.path)
        width, height = image.size
        if mode == FIT:
            return adjust_crop(dst_x, dst_y, width, height)