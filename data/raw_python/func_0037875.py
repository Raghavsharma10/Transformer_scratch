def create_tar(self):
        """Create a tar file with all the files."""

        def add_file_to_tar(tar, orig_fn, new_fn, func=None):
            tf = tarfile.TarInfo(name=new_fn)
            with open(orig_fn) as f:
                tfs = f.read()

            if func is not None:
                tfs = func(tfs)
            tf.size = len(tfs)
            tfs = io.BytesIO(tfs.encode('utf8'))
            tar.addfile(tarinfo=tf, fileobj=tfs)

        def add_text_to_tar(tar, new_fn, text, func=None):
            tf = tarfile.TarInfo(name=new_fn)
            if func is not None:
                text = func(text)
            tf.size = len(text)
            tfs = io.BytesIO(text.encode('utf8'))
            tar.addfile(tarinfo=tf, fileobj=tfs)

        def strip_lines(text):
            text = text.replace("\t", " ")
            while text.find("  ") != -1:
                text = text.replace("  ", " ")
            lines = [x.strip() for x in text.strip().split("\n")]
            return "\n".join(lines) + "\n"

        tar = tarfile.TarFile(self._tar_fn, "w")
        for i in range(len(self.bams)):
            roc_fn = self.bams[i].roc_fn()
            t_roc_fn = os.path.basename(roc_fn)

            gp_fn = self.bams[i].gp_fn()
            t_gp_fn = os.path.basename(gp_fn)

            svg_fn = self.bams[i].svg_fn()
            t_svg_fn = os.path.basename(svg_fn)

            add_file_to_tar(tar, roc_fn, t_roc_fn)
            add_file_to_tar(
                tar, gp_fn, t_gp_fn, lambda x: strip_lines(x.replace(roc_fn, t_roc_fn).replace(svg_fn, t_svg_fn))
            )

        gp_fn = self._gp_fn
        t_gp_fn = os.path.basename(gp_fn)
        svg_dir = os.path.join(self.panel_dir, "graphics") + "/"
        roc_dir = os.path.join(self.panel_dir, "roc") + "/"
        add_file_to_tar(tar, gp_fn, t_gp_fn, lambda x: strip_lines(x.replace(svg_dir, "").replace(roc_dir, "")))

        makefile = [
            ".PHONY: all",
            "all:",
            "\tgnuplot *.gp",
            "clean:",
            "\trm -f *.svg",
            "",
        ]
        add_text_to_tar(tar, "Makefile", "\n".join(makefile))