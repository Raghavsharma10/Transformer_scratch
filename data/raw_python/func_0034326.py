def update_frame_attributes(self, attrib):
        """ For positioning update the frame """

        if "align" in self.user_defined:
            align = self.user_defined["align"]
            if "top" in align:
                attrib["style:vertical-pos"] = "top"
            if "right" in align:
                attrib["style:horizontal-pos"] = "right"
        return attrib