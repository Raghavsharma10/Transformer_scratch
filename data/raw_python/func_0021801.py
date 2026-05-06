def backup(mongo_username, mongo_password, local_backup_directory_path, database=None,
           attached_directory_path=None, custom_prefix="backup",
           mongo_backup_directory_path="/tmp/mongo_dump",
           s3_bucket=None, s3_access_key_id=None, s3_secret_key=None,
           purge_local=None, purge_attached=None, cleanup=True, silent=False):
    """
    Runs a backup operation to At Least a local directory.
    You must provide mongodb credentials along with a a directory for a dump
    operation and a directory to contain your compressed backup.
    
        backup_prefix: optionally provide a prefix to be prepended to your backups,
            by default the prefix is "backup".
        database: optionally provide the name of one specific database to back up
            (instead of backing up all databases on the MongoDB server)
        attached_directory_path: makes a second copy of the backup to a different
            directory.  This directory is checked before other operations and
            will raise an error if it cannot be found.
        s3_bucket: if you have an Amazon Web Services S3 account you can
            automatically upload the backup to an S3 Bucket you provide;
            requires s3_access_key_id and s3_secret key to be passed as well
        s3_access_key_id, s3_secret_key: credentials for your AWS account.
        purge_local: An integer value, the number of days of backups to purge
            from local_backup_directory_path after operations have completed.
        purge_attached: An integer value, the number of days of backups to purge
            from attached_directory_path after operations have completed.
        cleanup: set to False to leave the mongo_backup_directory_path after operations
            have completed.
    """
    
    if attached_directory_path:
        if not path.exists(attached_directory_path):
            raise Exception("ERROR.  Would have to create %s for your attached storage, make sure that file paths already exist and re-run"
                            % (attached_directory_path))
    
    # Dump mongo, tarbz, copy to attached storage, upload to s3, purge, clean.
    full_file_name_path = local_backup_directory_path + custom_prefix + time_string()
    mongodump(mongo_username, mongo_password, mongo_backup_directory_path, database, silent=silent)
    
    local_backup_file = tarbz(mongo_backup_directory_path, full_file_name_path, silent=silent)
    
    if attached_directory_path:
        copy(local_backup_file, attached_directory_path + local_backup_file.split("/")[-1])
    
    if s3_bucket:
        s3_upload(local_backup_file, s3_bucket, s3_access_key_id, s3_secret_key)
    
    if purge_local:
        purge_date = (datetime.utcnow().replace(second=0, microsecond=0) -
                       timedelta(days=purge_local))
        purge_old_files(purge_date, local_backup_directory_path, custom_prefix=custom_prefix)
    
    if purge_attached and attached_directory_path:
        purge_date = (datetime.utcnow().replace(second=0, microsecond=0) -
                       timedelta(days=purge_attached))
        purge_old_files(purge_date, attached_directory_path, custom_prefix=custom_prefix)
    
    if cleanup:
        rmtree(mongo_backup_directory_path)