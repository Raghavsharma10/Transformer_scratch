def image(self,path_img):
        """ Open image file """
        im_open = Image.open(path_img)
        im = im_open.convert("RGB")
        # Convert the RGB image in printable image
        pix_line, img_size = self._convert_image(im)
        self._print_image(pix_line, img_size)