def get_node(self):
        """
	    <anim:par smil:begin="next">
	      <anim:par smil:begin="0s">
		<anim:par smil:begin="0s" smil:fill="hold" presentation:node-type="on-click" presentation:preset-class="entrance" presentation:preset-id="ooo-entrance-appear">
		  <anim:set smil:begin="0s" smil:dur="0.001s" smil:fill="hold" smil:targetElement="id1" anim:sub-item="text" smil:attributeName="visibility" smil:to="visible"/>
		</anim:par>
	      </anim:par>
	    </anim:par>
        """
        par = el("anim:par", attrib={"smil:begin": "next"})
        par2 = sub_el(par, "anim:par", attrib={"smil:begin": "0s"})
        par3 = sub_el(
            par2,
            "anim:par",
            attrib={
                "smil:begin": "0s",
                "smil:fill": "hold",
                "presentation:node-type": "on-click",
                "presentation:preset-class": "entrance",
                "presentation:preset-id": "ooo-entrance-appear",
            },
        )
        if self.ids:
            for id in self.ids:
                sub_el(
                    par3,
                    "anim:set",
                    attrib={
                        "smil:begin": "0s",
                        "smil:dur": "0.001s",
                        "smil:fill": "hold",
                        "smil:targetElement": id,
                        "anim:sub-item": "text",
                        "smil:attributeName": "visibility",
                        "smil:to": "visible",
                    },
                )

        else:
            sub_el(
                par3,
                "anim:set",
                attrib={
                    "smil:begin": "0s",
                    "smil:dur": "0.001s",
                    "smil:fill": "hold",
                    "smil:targetElement": self.id,
                    "anim:sub-item": "text",
                    "smil:attributeName": "visibility",
                    "smil:to": "visible",
                },
            )

        return par