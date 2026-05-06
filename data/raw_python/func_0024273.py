def get_workflows(self):
        """
        Scans and loads all wf found under WORKFLOW_PACKAGES_PATHS

        Yields: XML content of diagram file

        """
        for pth in settings.WORKFLOW_PACKAGES_PATHS:
            for f in glob.glob("%s/*.bpmn" % pth):
                with open(f) as fp:
                    yield os.path.basename(os.path.splitext(f)[0]), fp.read()