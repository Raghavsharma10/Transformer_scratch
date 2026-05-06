def ls(args):
    """
    List status of all configured SNS-SQS message buses and instances subscribed to them.
    """
    table = []
    queues = list(resources.sqs.queues.filter(QueueNamePrefix="github"))
    max_age = datetime.now(tzutc()) - timedelta(days=15)
    for topic in resources.sns.topics.all():
        account_id = ARN(topic.arn).account_id
        try:
            bucket = resources.s3.Bucket("deploy-status-{}".format(account_id))
            status_objects = bucket.objects.filter(Prefix=ARN(topic.arn).resource)
            recent_status_objects = {o.key: o for o in status_objects if o.last_modified > max_age}
        except ClientError:
            continue
        if ARN(topic.arn).resource.startswith("github"):
            for queue in queues:
                queue_name = os.path.basename(queue.url)
                if queue_name.startswith(ARN(topic.arn).resource):
                    row = dict(Topic=topic, Queue=queue)
                    status_object = bucket.Object(os.path.join(queue_name, "status"))
                    if status_object.key not in recent_status_objects:
                        continue
                    try:
                        github, owner, repo, events, instance = os.path.dirname(status_object.key).split("-", 4)
                        status = json.loads(status_object.get()["Body"].read().decode("utf-8"))
                        row.update(status, Owner=owner, Repo=repo, Instance=instance,
                                   Updated=status_object.last_modified)
                    except Exception:
                        pass
                    table.append(row)
    args.columns = ["Owner", "Repo", "Instance", "Status", "Ref", "Commit", "Updated", "Topic", "Queue"]
    page_output(tabulate(table, args))