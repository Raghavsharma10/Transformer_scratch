def rename(args):
    """Supply two names: Existing instance name or ID, and new name to assign to the instance."""
    old_name, new_name = args.names
    add_tags(resources.ec2.Instance(resolve_instance_id(old_name)), Name=new_name, dry_run=args.dry_run)