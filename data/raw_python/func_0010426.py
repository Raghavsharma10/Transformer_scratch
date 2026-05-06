def report(self, violations):
        """
        Return readable report of accessibility violations found.

        :param violations: Dictionary of violations.
        :type violations: dict
        :return report: Readable report of violations.
        :rtype: string
        """
        string = ""
        string += "Found " + str(len(violations)) + " accessibility violations:"
        for violation in violations:
            string += (
                "\n\n\nRule Violated:\n"
                + violation["id"]
                + " - "
                + violation["description"]
                + "\n\tURL: "
                + violation["helpUrl"]
                + "\n\tImpact Level: "
                + violation["impact"]
                + "\n\tTags:"
            )
            for tag in violation["tags"]:
                string += " " + tag
            string += "\n\tElements Affected:"
            i = 1
            for node in violation["nodes"]:
                for target in node["target"]:
                    string += "\n\t" + str(i) + ") Target: " + target
                    i += 1
                for item in node["all"]:
                    string += "\n\t\t" + item["message"]
                for item in node["any"]:
                    string += "\n\t\t" + item["message"]
                for item in node["none"]:
                    string += "\n\t\t" + item["message"]
            string += "\n\n\n"
        return string