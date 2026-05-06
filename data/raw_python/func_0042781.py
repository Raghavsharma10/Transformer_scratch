def get_tuids(self, branch, revision, files):
        """
        GET TUIDS FROM ENDPOINT, AND STORE IN DB
        :param branch: BRANCH TO FIND THE REVISION/FILE
        :param revision: THE REVISION NUNMBER
        :param files: THE FULL PATHS TO THE FILES
        :return: MAP FROM FILENAME TO TUID LIST
        """

        # SCRUB INPUTS
        revision = revision[:12]
        files = [file.lstrip('/') for file in files]

        with Timer(
            "ask tuid service for {{num}} files at {{revision|left(12)}}",
            {"num": len(files), "revision": revision},
            silent=not DEBUG or not self.enabled
        ):
            response = self.db.query(
                "SELECT file, tuids FROM tuid WHERE revision=" + quote_value(revision) +
                " AND file IN " + quote_list(files)
            )
            found = {file: json2value(tuids) for file, tuids in response.data}

            try:
                remaining = set(files) - set(found.keys())
                new_response = None
                if remaining:
                    request = wrap({
                        "from": "files",
                        "where": {"and": [
                            {"eq": {"revision": revision}},
                            {"in": {"path": remaining}},
                            {"eq": {"branch": branch}}
                        ]},
                        "branch": branch,
                        "meta": {
                            "format": "list",
                            "request_time": Date.now()
                        }
                    })
                    if self.push_queue is not None:
                        if DEBUG:
                            Log.note("record tuid request to SQS: {{timestamp}}", timestamp=request.meta.request_time)
                        self.push_queue.add(request)
                    else:
                        if DEBUG:
                            Log.note("no recorded tuid request")

                    if not self.enabled:
                        return found

                    new_response = http.post_json(
                        self.endpoint,
                        json=request,
                        timeout=self.timeout
                    )

                    if new_response.data and any(r.tuids for r in new_response.data):
                        try:
                            with self.db.transaction() as transaction:


                                command = "INSERT INTO tuid (revision, file, tuids) VALUES " + sql_list(
                                    quote_list((revision, r.path, value2json(r.tuids)))
                                    for r in new_response.data
                                    if r.tuids != None
                                )
                                if not command.endswith(" VALUES "):
                                    transaction.execute(command)
                        except Exception as e:
                            Log.error("can not insert {{data|json}}", data=new_response.data, cause=e)
                self.num_bad_requests = 0

                found.update({r.path: r.tuids for r in new_response.data} if new_response else {})
                return found

            except Exception as e:
                self.num_bad_requests += 1
                if self.enabled:
                    if "502 Bad Gateway" in e:
                        self.enabled = False
                        Log.error("TUID service has problems.", cause=e)
                    elif self.num_bad_requests >= MAX_BAD_REQUESTS:
                        self.enabled = False
                        Log.error("TUID service has problems.", cause=e)
                    else:
                        Log.warning("TUID service has problems.", cause=e)
                        Till(seconds=SLEEP_ON_ERROR).wait()
                return found