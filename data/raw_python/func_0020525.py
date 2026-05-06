def create_build_from_buildrequest(self, build_request):
        """
        render provided build_request and submit build from it

        :param build_request: instance of build.build_request.BuildRequest
        :return: instance of build.build_response.BuildResponse
        """
        build_request.set_openshift_required_version(self.os_conf.get_openshift_required_version())
        build = build_request.render()
        response = self.os.create_build(json.dumps(build))
        build_response = BuildResponse(response.json(), self)
        return build_response