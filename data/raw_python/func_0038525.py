def read(cls, string):
        """Returns dict with separated frontmatter from string.

        Returned dict keys:
        attributes -- extracted YAML attributes in dict form.
        body -- string contents below the YAML separators
        frontmatter -- string representation of YAML
        """
        fmatter = ""
        body = ""
        result = cls._regex.search(string)

        if result:
            fmatter = result.group(1)
            body = result.group(2)
        return {
            "attributes": yaml.load(fmatter),
            "body": body,
            "frontmatter": fmatter,
        }