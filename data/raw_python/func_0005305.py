def main():
    """Handles calling this module as a script

    :return: None
    """
    log = logging.getLogger(mod_logger + '.main')
    parser = argparse.ArgumentParser(description='This Python module retrieves artifacts from Nexus.')
    parser.add_argument('-u', '--url', help='Nexus Server URL', required=False)
    parser.add_argument('-g', '--groupId', help='Group ID', required=True)
    parser.add_argument('-a', '--artifactId', help='Artifact ID', required=True)
    parser.add_argument('-v', '--version', help='Artifact Version', required=True)
    parser.add_argument('-c', '--classifier', help='Artifact Classifier', required=False)
    parser.add_argument('-p', '--packaging', help='Artifact Packaging', required=True)
    parser.add_argument('-r', '--repo', help='Nexus repository name', required=False)
    parser.add_argument('-d', '--destinationDir', help='Directory to download to', required=True)
    parser.add_argument('-n', '--username', help='Directory to download to', required=True)
    parser.add_argument('-w', '--password', help='Directory to download to', required=True)
    args = parser.parse_args()
    try:
        get_artifact(
            nexus_url=args.url,
            group_id=args.groupId,
            artifact_id=args.artifactId,
            version=args.version,
            classifier=args.classifier,
            packaging=args.packaging,
            repo=args.repo,
            destination_dir=args.destinationDir,
            username=args.username,
            password=args.password
        )
    except Exception as e:
        msg = 'Caught exception {n}, unable for download artifact from Nexus\n{s}'.format(
            n=e.__class__.__name__, s=e)
        log.error(msg)
        return