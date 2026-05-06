def skip_build(self):
        """Check if build should be skipped
        """
        skip_msg = self.config.get('skip', '[ci skip]')
        return (
            os.environ.get('CODEBUILD_BUILD_SUCCEEDING') == '0' or
            self.info['current_tag'] or
            skip_msg in self.info['head']['message']
        )