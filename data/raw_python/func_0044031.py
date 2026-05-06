def get_time_buckets_from_metadata(metadata):
        '''return a list of time buckets in which the metadata falls'''
        start = metadata['start']
        end = metadata.get('end') or start
        buckets = DatalakeRecord.get_time_buckets(start, end)
        if len(buckets) > DatalakeRecord.MAXIMUM_BUCKET_SPAN:
            msg = 'metadata spans too many time buckets: {}'
            j = json.dumps(metadata)
            msg = msg.format(j)
            raise UnsupportedTimeRange(msg)
        return buckets