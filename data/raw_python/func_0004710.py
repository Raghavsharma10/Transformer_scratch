def _parse_gist(gist):
        """Receive a gist (dict) and parse it to GetGist"""

        # parse files
        files = list()
        file_names = sorted(filename for filename in gist["files"].keys())
        for name in file_names:
            files.append(
                dict(filename=name, raw_url=gist["files"][name].get("raw_url"))
            )

        # parse description
        description = gist["description"]
        if not description:
            names = sorted(f.get("filename") for f in files)
            description = names.pop(0)

        return dict(
            description=description,
            id=gist.get("id"),
            files=files,
            url=gist.get("html_url"),
        )