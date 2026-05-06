def run_osa_differ():
    """Start here."""
    # Get our arguments from the command line
    args = parse_arguments()

    # Set up DEBUG logging if needed
    if args.debug:
        log.setLevel(logging.DEBUG)
    elif args.verbose:
        log.setLevel(logging.INFO)

    # Create the storage directory if it doesn't exist already.
    try:
        storage_directory = prepare_storage_dir(args.directory)
    except OSError:
        print("ERROR: Couldn't create the storage directory {0}. "
              "Please create it manually.".format(args.directory))
        sys.exit(1)

    # Assemble some variables for the OSA repository.
    osa_old_commit = args.old_commit[0]
    osa_new_commit = args.new_commit[0]
    osa_repo_dir = "{0}/openstack-ansible".format(storage_directory)

    # Generate OpenStack-Ansible report header.
    report_rst = make_osa_report(osa_repo_dir,
                                 osa_old_commit,
                                 osa_new_commit,
                                 args)

    # Get OpenStack-Ansible Reno release notes for the packaged
    # releases between the two commits.
    if args.release_notes:
        report_rst += ("\nRelease Notes\n"
                       "-------------")
        report_rst += get_release_notes(osa_repo_dir,
                                        osa_old_commit,
                                        osa_new_commit)

    # Get the list of OpenStack roles from the newer and older commits.
    role_yaml = get_roles(osa_repo_dir,
                          osa_old_commit,
                          args.role_requirements)
    role_yaml_latest = get_roles(osa_repo_dir,
                                 osa_new_commit,
                                 args.role_requirements)

    if not args.skip_roles:
        # Generate the role report.
        report_rst += ("\nOpenStack-Ansible Roles\n"
                       "-----------------------")
        report_rst += make_report(storage_directory,
                                  role_yaml,
                                  role_yaml_latest,
                                  args.update,
                                  args.version_mappings)

    if not args.skip_projects:
        # Get the list of OpenStack projects from newer commit and older
        # commit.
        project_yaml = get_projects(osa_repo_dir, osa_old_commit)
        project_yaml_latest = get_projects(osa_repo_dir,
                                           osa_new_commit)

        # Generate the project report.
        report_rst += ("\nOpenStack Projects\n"
                       "------------------")
        report_rst += make_report(storage_directory,
                                  project_yaml,
                                  project_yaml_latest,
                                  args.update)

    # Publish report according to the user's request.
    output = publish_report(report_rst, args, osa_old_commit, osa_new_commit)
    print(output)