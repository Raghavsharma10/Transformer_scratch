def list_builds(self, field_selector=None, koji_task_id=None, running=None,
                    labels=None):
        """
        List builds with matching fields

        :param field_selector: str, field selector for Builds
        :param koji_task_id: str, only list builds for Koji Task ID
        :return: BuildResponse list
        """

        if running:
            running_fs = ",".join(["status!={status}".format(status=status.capitalize())
                                  for status in BUILD_FINISHED_STATES])
            if not field_selector:
                field_selector = running_fs
            else:
                field_selector = ','.join([field_selector, running_fs])
        response = self.os.list_builds(field_selector=field_selector,
                                       koji_task_id=koji_task_id, labels=labels)
        serialized_response = response.json()
        build_list = []
        for build in serialized_response["items"]:
            build_list.append(BuildResponse(build, self))

        return build_list