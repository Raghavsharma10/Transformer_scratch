def add_otp_style(self, zip_odp, style_file):
        """
        takes the slide content and merges in the style_file
        """
        style = zipwrap.Zippier(style_file)
        for picture_file in style.ls("Pictures"):
            zip_odp.write(picture_file, style.cat(picture_file, True))
        xml_data = style.cat("styles.xml", False)
        # import pdb;pdb.set_trace()
        xml_data = self.override_styles(xml_data)
        zip_odp.write("styles.xml", xml_data)