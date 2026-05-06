def extend(self, records):
        """
        records - MUST HAVE FORM OF
            [{"value":value}, ... {"value":value}] OR
            [{"json":json}, ... {"json":json}]
            OPTIONAL "id" PROPERTY IS ALSO ACCEPTED
        """
        if self.settings.read_only:
            Log.error("Index opened in read only mode, no changes allowed")
        lines = []
        try:
            for r in records:
                if '_id' in r or 'value' not in r:  # I MAKE THIS MISTAKE SO OFTEN, I NEED A CHECK
                    Log.error('Expecting {"id":id, "value":document} form.  Not expecting _id')
                id, version, json_bytes = self.encode(r)
                if '"_id":' in json_bytes:
                    id, version, json_bytes = self.encode(r)

                if version:
                    lines.append(value2json({"index": {"_id": id, "version": int(version), "version_type": "external_gte"}}))
                else:
                    lines.append('{"index":{"_id": ' + value2json(id) + '}}')
                lines.append(json_bytes)

            del records

            if not lines:
                return

            with Timer("Add {{num}} documents to {{index}}", {"num": int(len(lines) / 2), "index": self.settings.index}, silent=not self.debug):
                try:
                    data_string = "\n".join(l for l in lines) + "\n"
                except Exception as e:
                    raise Log.error("can not make request body from\n{{lines|indent}}", lines=lines, cause=e)

                wait_for_active_shards = coalesce(
                    self.settings.wait_for_active_shards,
                    {"one": 1, None: None}[self.settings.consistency]
                )

                response = self.cluster.post(
                    self.path + "/_bulk",
                    data=data_string,
                    headers={"Content-Type": "application/x-ndjson"},
                    timeout=self.settings.timeout,
                    retry=self.settings.retry,
                    params={"wait_for_active_shards": wait_for_active_shards}
                )
                items = response["items"]

                fails = []
                if self.cluster.version.startswith("0.90."):
                    for i, item in enumerate(items):
                        if not item.index.ok:
                            fails.append(i)
                elif self.cluster.version.startswith(("1.4.", "1.5.", "1.6.", "1.7.", "5.", "6.")):
                    for i, item in enumerate(items):
                        if item.index.status == 409:  # 409 ARE VERSION CONFLICTS
                            if "version conflict" not in item.index.error.reason:
                                fails.append(i)  # IF NOT A VERSION CONFLICT, REPORT AS FAILURE
                        elif item.index.status not in [200, 201]:
                            fails.append(i)
                else:
                    Log.error("version not supported {{version}}", version=self.cluster.version)

                if fails:
                    if len(fails) <= 3:
                        cause = [
                            Except(
                                template="{{status}} {{error}} (and {{some}} others) while loading line id={{id}} into index {{index|quote}} (typed={{typed}}):\n{{line}}",
                                params={
                                    "status":items[i].index.status,
                                    "error":items[i].index.error,
                                    "some":len(fails) - 1,
                                    "line":strings.limit(lines[i * 2 + 1], 500 if not self.debug else 100000),
                                    "index":self.settings.index,
                                    "typed":self.settings.typed,
                                    "id":items[i].index._id
                                }
                            )
                            for i in fails
                        ]
                    else:
                        i=fails[0]
                        cause = Except(
                            template="{{status}} {{error}} (and {{some}} others) while loading line id={{id}} into index {{index|quote}} (typed={{typed}}):\n{{line}}",
                            params={
                                "status":items[i].index.status,
                                "error":items[i].index.error,
                                "some":len(fails) - 1,
                                "line":strings.limit(lines[i * 2 + 1], 500 if not self.debug else 100000),
                                "index":self.settings.index,
                                "typed":self.settings.typed,
                                "id":items[i].index._id
                            }
                        )
                    Log.error("Problems with insert", cause=cause)
            pass
        except Exception as e:
            e = Except.wrap(e)
            if e.message.startswith("sequence item "):
                Log.error("problem with {{data}}", data=text_type(repr(lines[int(e.message[14:16].strip())])), cause=e)
            Log.error("problem sending to ES", cause=e)