def deploy(www_dir, bucket_name):
    """ Deploy to the configured S3 bucket. """

    # Set up the connection to an S3 bucket.
    conn = boto.connect_s3()
    bucket = conn.get_bucket(bucket_name)

    # Deploy each changed file in www_dir
    os.chdir(www_dir)
    for root, dirs, files in os.walk('.'):
        for f in files:
            # Use full relative path. Normalize to remove dot.
            file_path = os.path.normpath(os.path.join(root, f))

            if has_changed_since_last_deploy(file_path, bucket):
                deploy_file(file_path, bucket)
            else:
                logger.info("Skipping {0}".format(file_path))

    # Make the whole bucket public
    bucket.set_acl('public-read')

    # Configure it to be a website
    bucket.configure_website('index.html', 'error.html')

    # Print the endpoint, so you know the URL
    msg = "Your website is now live at {0}".format(
        bucket.get_website_endpoint())
    logger.info(msg)
    logger.info("If you haven't done so yet, point your domain name there!")