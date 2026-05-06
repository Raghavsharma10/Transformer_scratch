def show_images(self, size="small"):
        """Shows preview images using the Jupyter notebook HTML display.

        Parameters
        ==========
        size : {'small', 'med', 'thumb', 'full'}
            Determines the size of the preview image to be shown.
        """
        d = dict(small=256, med=512, thumb=100, full=1024)
        try:
            width = d[size]
        except KeyError:
            print("Allowed keys:", d.keys())
            return
        img_urls = [i._get_img_url(size) for i in self.obsids]
        imagesList = "".join(
            [
                "<img style='width: {0}px; margin: 0px; float: "
                "left; border: 1px solid black;' "
                "src='{1}' />".format(width, s)
                for s in img_urls
            ]
        )
        display(HTML(imagesList))