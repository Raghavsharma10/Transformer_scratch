def _add_styles(self, add_paragraph=True, add_text=True):
        """
        Adds paragraph and span wrappers if necessary based on style
        """
        p_styles = self.get_para_styles()
        t_styles = self.get_span_styles()
        for s in self.slide.pending_styles:
            if isinstance(s, ParagraphStyle):
                p_styles.update(s.styles)
            elif isinstance(s, TextStyle):
                t_styles.update(s.styles)

        para = ParagraphStyle(**p_styles)

        if add_paragraph or self.slide.paragraph_attribs:
            p_attrib = {ns("text", "style-name"): para.name}
            p_attrib.update(self.slide.paragraph_attribs)
            if not self._in_tag(ns("text", "p"), p_attrib):
                self.parent_of(ns("text", "p"))
                # Create paragraph style first
                self.slide._preso.add_style(para)
                self.add_node("text:p", attrib=p_attrib)

        # span is only necessary if style changes
        if add_text and t_styles:
            text = TextStyle(**t_styles)
            children = self.cur_node.getchildren()
            if children:
                # if we already are using this text style, reuse the last one
                last = children[-1]
                if (
                    last.tag == ns("text", "span")
                    and last.attrib[ns("text", "style-name")] == text.name
                    and last.tail is None
                ):  # if we have a tail, we can't reuse
                    self.cur_node = children[-1]
                    return

            if not self._is_node(
                ns("text", "span"), {ns("text", "style-name"): text.name}
            ):
                # Create text style
                self.slide._preso.add_style(text)
                self.add_node("text:span", attrib={"text:style-name": text.name})