def handle(self, *args, **options):
        """This function is called by the Django API to specify how this object
        will be saved to the database.
        """
        taxonomy_id = options['taxonomy_id']

        # Remove leading and trailing blank characters in "common_name"
        # and "scientific_name
        common_name = options['common_name'].strip()
        scientific_name = options['scientific_name'].strip()

        if common_name and scientific_name:
            # A 'slug' is a label for an object in django, which only contains
            # letters, numbers, underscores, and hyphens, thus making it URL-
            # usable.  The slugify method in django takes any string and
            # converts it to this format.  For more information, see:
            # http://stackoverflow.com/questions/427102/what-is-a-slug-in-django
            slug = slugify(scientific_name)
            logger.info("Slug generated: %s", slug)

            # If organism exists, update with passed parameters
            try:
                org = Organism.objects.get(taxonomy_id=taxonomy_id)
                org.common_name = common_name
                org.scientific_name = scientific_name
                org.slug = slug
            # If organism doesn't exist, construct an organism object
            # (see organisms/models.py).
            except Organism.DoesNotExist:
                org = Organism(taxonomy_id=taxonomy_id,
                               common_name=common_name,
                               scientific_name=scientific_name,
                               slug=slug
                )
            org.save()  # Save to the database.
        else:
            # Report an error if the user did not fill out all fields.
            logger.error(
                "Failed to add or update organism. "
                "Please check that all fields are filled correctly."
            )