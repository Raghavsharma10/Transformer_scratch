def _calc_positions(center, halfdim, border_width,
                        orientation, transforms):
        """
        Calculate the text centeritions given the ColorBar
        parameters.

        Note
        ----
        This is static because in principle, this
        function does not need access to the state of the ColorBar
        at all. It's a computation function that computes coordinate
        transforms

        Parameters
        ----------
        center: tuple (x, y)
            Center of the ColorBar
        halfdim: tuple (halfw, halfh)
            Half of the dimensions measured from the center
        border_width: float
            Width of the border of the ColorBar
        orientation: "top" | "bottom" | "left" | "right"
            Position of the label with respect to the ColorBar
        transforms: TransformSystem
            the transforms of the ColorBar
        """
        (x, y) = center
        (halfw, halfh) = halfdim

        visual_to_doc = transforms.get_transform('visual', 'document')
        doc_to_visual = transforms.get_transform('document', 'visual')

        # doc_widths = visual_to_doc.map(np.array([halfw, halfh, 0, 0],
        #                                         dtype=np.float32))

        doc_x = visual_to_doc.map(np.array([halfw, 0, 0, 0], dtype=np.float32))
        doc_y = visual_to_doc.map(np.array([0, halfh, 0, 0], dtype=np.float32))

        if doc_x[0] < 0:
            doc_x *= -1

        if doc_y[1] < 0:
            doc_y *= -1

        # doc_halfw = np.abs(doc_widths[0])
        # doc_halfh = np.abs(doc_widths[1])

        if orientation == "top":
            doc_perp_vector = -doc_y
        elif orientation == "bottom":
            doc_perp_vector = doc_y
        elif orientation == "left":
            doc_perp_vector = -doc_x
        if orientation == "right":
            doc_perp_vector = doc_x

        perp_len = np.linalg.norm(doc_perp_vector)
        doc_perp_vector /= perp_len
        perp_len += border_width
        perp_len += 5  # pixels
        perp_len *= ColorBarVisual.text_padding_factor
        doc_perp_vector *= perp_len

        doc_center = visual_to_doc.map(np.array([x, y, 0, 0],
                                                dtype=np.float32))
        doc_label_pos = doc_center + doc_perp_vector
        visual_label_pos = doc_to_visual.map(doc_label_pos)[:3]

        # next, calculate tick positions
        if orientation in ["top", "bottom"]:
            doc_ticks_pos = [doc_label_pos - doc_x,
                             doc_label_pos + doc_x]
        else:
            doc_ticks_pos = [doc_label_pos + doc_y,
                             doc_label_pos - doc_y]

        visual_ticks_pos = []
        visual_ticks_pos.append(doc_to_visual.map(doc_ticks_pos[0])[:3])
        visual_ticks_pos.append(doc_to_visual.map(doc_ticks_pos[1])[:3])

        return (visual_label_pos, visual_ticks_pos)