def get_prepare_by_id(
            self,
            prepare_id=None):
        """get_prepare_by_id

        :param prepare_id: MLJob.id in the database
        """

        if not prepare_id:
            log.error("missing prepare_id for get_prepare_by_id")
            return self.build_response(
                status=ERROR,
                error="missing prepare_id for get_prepare_by_id")

        if self.debug:
            log.info(("user={} getting prepare={}")
                     .format(
                        self.user,
                        prepare_id))

        url = "{}{}".format(
                self.api_urls["prepare"],
                prepare_id)

        not_done = True
        while not_done:

            if self.debug:
                log.info((
                    "JOB attempting to get={} to url={} "
                    "verify={} cert={}").format(
                        prepare_id,
                        url,
                        self.use_verify,
                        self.cert))

            response = requests.get(
                url,
                verify=self.use_verify,
                cert=self.cert,
                headers=self.get_auth_header())

            if self.debug:
                log.info(("JOB response status_code={} text={} reason={}")
                         .format(
                            response.status_code,
                            response.text,
                            response.reason))

            if response.status_code == 401:
                login_res = self.retry_login()
                if login_res["status"] != SUCCESS:
                    if self.verbose:
                        log.error(
                            "retry login attempts failed")
                    return self.build_response(
                        status=login_res["status"],
                        error=login_res["error"])
                # if able to log back in just retry the call
            elif response.status_code == 200:

                if self.verbose:
                    log.debug("deserializing")

                prepare_data = json.loads(
                    response.text)

                prepare_id = prepare_data.get(
                    "id",
                    None)

                if not prepare_id:
                    return self.build_response(
                        status=ERROR,
                        error="missing prepare.id",
                        data="text={} reason={}".format(
                            response.reason,
                            response.text))

                self.all_prepares[str(prepare_id)] = prepare_data

                if self.debug:
                    log.info(("added prepare={} all_prepares={}")
                             .format(
                                prepare_id,
                                len(self.all_prepares)))

                return self.build_response(
                    status=SUCCESS,
                    error="",
                    data=prepare_data)
            else:
                err_msg = ("failed with "
                           "status_code={} text={} reason={}").format(
                               response.status_code,
                               response.text,
                               response.reason)
                if self.verbose:
                    log.error(err_msg)
                return self.build_response(
                    status=ERROR,
                    error=err_msg)