def grant(args):
    """
    Given an IAM role or instance name, attach an IAM policy granting
    appropriate permissions to subscribe to deployments. Given a
    GitHub repo URL, create and record deployment keys for the repo
    and any of its private submodules, making the keys accessible to
    the IAM role.
    """
    try:
        role = resources.iam.Role(args.iam_role_or_instance)
        role.load()
    except ClientError:
        role = get_iam_role_for_instance(args.iam_role_or_instance)
    role.attach_policy(PolicyArn=ensure_deploy_iam_policy().arn)
    for private_repo in [args.repo] + list(private_submodules(args.repo)):
        gh_owner_name, gh_repo_name = parse_repo_name(private_repo)
        secret = secrets.put(argparse.Namespace(secret_name="deploy.{}.{}".format(gh_owner_name, gh_repo_name),
                                                iam_role=role.name,
                                                instance_profile=None,
                                                iam_group=None,
                                                iam_user=None,
                                                generate_ssh_key=True))
        get_repo(private_repo).create_key(__name__ + "." + role.name, secret["ssh_public_key"])
        logger.info("Created deploy key %s for IAM role %s to access GitHub repo %s",
                    secret["ssh_key_fingerprint"], role.name, private_repo)