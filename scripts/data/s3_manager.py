import json
import urllib.parse
import boto3
import awswrangler as wr


class S3Manager:
    def __init__(self):
        self.s3 = boto3.resource('s3')
        self.path_template = "s3://{bucket}/{key}/{filename}"

    def save_parquet(self, df, bucket, key, filename, cols=[]):
        path = self.path_template.format(bucket=bucket, key=key, filename=filename)
        wr.s3.to_parquet(
            df = df,
            path = path,
            partition_cols = cols,
            dataset = True,
            mode = "overwrite_partitions"
        )

     
