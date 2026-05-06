def __update(self):
        """
        This is called each time an attribute is asked, to be sure every params are updated, beceause of callbacks.
        """

        # I can not set the size attr because it is my property, so I set the width and height separately
        width, height = self.size
        super(BaseWidget, self).__setattr__("width", width)
        super(BaseWidget, self).__setattr__("height", height)
        super(BaseWidget, self).__setattr__(self.anchor, self.pos)