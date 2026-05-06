def visual_border_width(self):
        """ The border width in visual coordinates
        """
        render_to_doc =  \
            self.transforms.get_transform('document', 'visual')

        vec = render_to_doc.map([self.border_width, self.border_width, 0])
        origin = render_to_doc.map([0, 0, 0])

        visual_border_width = [vec[0] - origin[0], vec[1] - origin[1]]

        # we need to flip the y axis because coordinate systems are inverted
        visual_border_width[1] *= -1

        return visual_border_width