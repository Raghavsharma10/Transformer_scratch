def get_node(self):
        """return etree Element representing this slide"""
        # already added title, text frames
        # add animation chunks
        if self.animations:
            anim_par = el("anim:par", attrib={"presentation:node-type": "timing-root"})
            self._page.append(anim_par)
            anim_seq = sub_el(
                anim_par, "anim:seq", attrib={"presentation:node-type": "main-sequence"}
            )
            for a in self.animations:
                a_node = a.get_node()
                anim_seq.append(a_node)

        # add notes now (so they are last)
        if self.notes_frame:
            notes = self.notes_frame.get_node()
            self._page.append(notes)
        if self.footer:
            self._page.attrib[ns("presentation", "use-footer-name")] = self.footer.name
        return self._page