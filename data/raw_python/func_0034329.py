def add_picture(self, p):
        """
        needs to look like this (under draw:page)

	<draw:frame draw:style-name="gr2" draw:text-style-name="P2" draw:layer="layout" svg:width="19.589cm" svg:height="13.402cm" svg:x="3.906cm" svg:y="4.378cm">
	  <draw:image xlink:href="Pictures/10000201000002F800000208188B22AE.png" xlink:type="simple" xlink:show="embed" xlink:actuate="onLoad">
	    <text:p text:style-name="P1"/>
	  </draw:image>
	</draw:frame>
        """
        # pictures should be added the the draw:frame element
        self.pic_frame = PictureFrame(self, p)
        self.pic_frame.add_node(
            "draw:image",
            attrib={
                "xlink:href": "Pictures/" + p.internal_name,
                "xlink:type": "simple",
                "xlink:show": "embed",
                "xlink:actuate": "onLoad",
            },
        )
        self._preso._pictures.append(p)
        node = self.pic_frame.get_node()
        self._page.append(node)