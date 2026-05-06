def __parse_app_list(app_list):
        """Parse list of apps for arguments.

        :param app_list: list of apps with optional arguments.
        :return: list of apps and assigned argument dict.
        :rtype: [String], {String: [String]}
        """
        args = {}
        apps = []
        for app_str in app_list:
            parts = app_str.split("&")
            app_path = parts[0].strip()
            apps.append(app_path)
            if len(parts) > 1:
                args[app_path] = [arg.strip() for arg in parts[1].split()]
            else:
                args[app_path] = []
        return apps, args