def _execute(self, app_, file_):
        """Run app with file as input.

        :param app_: application to run.
        :param file_: file to run app with.
        :return: success True, else False
        :rtype: bool
        """
        app_name = os.path.basename(app_)
        args = [app_]
        args.extend(self.args[app_])
        args.append(file_)
        process = subprocess.Popen(args)

        time.sleep(1)
        status = {True: Status.SUCCESS, False: Status.FAILED}
        crashed = process.poll()
        result = status[crashed is None]
        self.stats_.add(app_name, result)
        if result is Status.SUCCESS:
            # process did not crash, so just terminate it
            process.terminate()