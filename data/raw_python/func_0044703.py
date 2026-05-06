def _atexit_callback(cls):
        """Create Python script copies with updated baselines.

        For any baseline that had a miscompare, make a copy of the
        source file which contained the baseline and update the
        baseline with the new string value.

        :returns:
            record of every Python file update (key=path,
            value=script instance)
        :rtype: dict

        """
        updated_scripts = {}

        for baseline in cls._baselines_to_update:

            if baseline.z__path.endswith('<stdin>'):
                continue

            try:
                script = updated_scripts[baseline.z__path]
            except KeyError:
                script = Script(baseline.z__path)
                updated_scripts[baseline.z__path] = script

            script.add_update(baseline.z__linenum, baseline.z__update)

        for key in sorted(updated_scripts):
            script = updated_scripts[key]
            script.update()

        return updated_scripts