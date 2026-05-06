def screenshot(self, scale=None, quality=None):
        """
        Take a screenshot of device and log in the report with timestamp, scale for screenshot size and quality for screenshot quality
        default scale=1.0 quality=100
        """
        output_dir = BuiltIn().get_variable_value('${OUTPUTDIR}')
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
        screenshot_path = '%s%s%s.png' % (output_dir, os.sep, st)
        self.device.screenshot(screenshot_path, scale, quality)
        logger.info('\n<a href="%s">%s</a><br><img src="%s">' % (screenshot_path, st, screenshot_path), html=True)