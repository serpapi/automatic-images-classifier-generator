## BEFORE YOU RUN THIS FILE, YOU NEED TO MANUALLY CREATE A BUCKET CALLED images
from couchbase.options import (ClusterOptions, ClusterTimeoutOptions, QueryOptions)
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from datetime import timedelta
from credentials import *

username = SERVER_USERNAME
password = SERVER_PASSWORD
bucket_name = "images"
auth = PasswordAuthenticator(
	  username,
		password
)

timeout_opts = ClusterTimeoutOptions(kv_timeout=timedelta(seconds=10))
cluster = Cluster('couchbase://localhost', ClusterOptions(auth, timeout_options=timeout_opts))
cluster.wait_until_ready(timedelta(seconds=5))

queries = [ "CREATE SCOPE `images`.image;",
            "CREATE SCOPE `images`.model;",
            "CREATE COLLECTION `images`.image.labelled_image;",
            "CREATE COLLECTION `images`.model.attempt;",
            "CREATE PRIMARY INDEX ON `images`.image.labelled_image;",
            "CREATE PRIMARY INDEX ON `images`.model.attempt;"
]

for query in queries:
  row_iter = cluster.query(query, QueryOptions())
  for row in row_iter:
    print(row)

