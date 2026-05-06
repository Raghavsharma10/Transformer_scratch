def forwards(self, orm):
        "Write your forwards methods here."

        rows = db.execute("select distinct feature_type from content_content")
        for row in rows:
            feature_type = row[0]
            try:
                ft = orm.FeatureType.objects.get(slug=slugify(feature_type))
            except orm.FeatureType.DoesNotExist:
                ft = orm.FeatureType.objects.create(
                    name=feature_type,
                    slug=slugify(feature_type)
                )
            db.execute("update content_content set feature_type_id = %s where feature_type = %s", [ft.id, feature_type])