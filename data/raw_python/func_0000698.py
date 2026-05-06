def perform_async_refresh(cls, klass_str, obj_args, obj_kwargs, call_args, call_kwargs):
        """
        Re-populate cache using the given job class.

        The job class is instantiated with the passed constructor args and the
        refresh method is called with the passed call args.  That is::

            data = klass(*obj_args, **obj_kwargs).refresh(
                *call_args, **call_kwargs)

        :klass_str: String repr of class (eg 'apps.twitter.jobs.FetchTweetsJob')
        :obj_args: Constructor args
        :obj_kwargs: Constructor kwargs
        :call_args: Refresh args
        :call_kwargs: Refresh kwargs
        """
        klass = get_job_class(klass_str)
        if klass is None:
            logger.error("Unable to construct %s with args %r and kwargs %r",
                         klass_str, obj_args, obj_kwargs)
            return

        logger.info("Using %s with constructor args %r and kwargs %r",
                    klass_str, obj_args, obj_kwargs)
        logger.info("Calling refresh with args %r and kwargs %r", call_args,
                    call_kwargs)
        start = time.time()
        try:
            klass(*obj_args, **obj_kwargs).refresh(
                *call_args, **call_kwargs)
        except Exception as e:
            logger.exception("Error running job: '%s'", e)
        else:
            duration = time.time() - start
            logger.info("Refreshed cache in %.6f seconds", duration)