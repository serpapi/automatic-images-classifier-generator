from couchbase.options import (ClusterOptions, ClusterTimeoutOptions, QueryOptions)
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from serpapi import GoogleSearch
from datetime import timedelta
from pydantic import BaseModel
from typing import List
import mimetypes
import requests
import random
import uuid
import base64
from credentials import *

class Document(BaseModel):
	type: str = "image"
	id: str
	classification: str
	base64: str
	uri: str

class Attempt(BaseModel):
	id: int | None = None
	name: str | None = None
	training_commands: dict = {}
	training_losses: list = []
	n_epoch: int = 0
	testing_commands: dict = {}
	accuracy: float = 0.0
	status: str = "incomplete"
	limit: int = 0

class ModelsDatabase:
	def __init__(self):
		username = SERVER_USERNAME
		password = SERVER_PASSWORD
		bucket_name = "images"
		auth = PasswordAuthenticator(
				username,
				password
		)
		timeout_opts = ClusterTimeoutOptions(kv_timeout=timedelta(seconds=10))
		self.cluster = Cluster('couchbase://localhost', ClusterOptions(auth, timeout_options=timeout_opts))
		self.cluster.wait_until_ready(timedelta(seconds=5))
		cb = self.cluster.bucket(bucket_name)
		self.cb_coll = cb.scope("model").collection("attempt")
	
	def insert_attempt(self, doc: Attempt):
		doc = doc.dict()
		print("\nInsert CAS: ")
		try:
			key = doc["name"]
			result = self.cb_coll.insert(key, doc)
			print(result.cas)
		except Exception as e:
			print(e)

	def get_attempt_by_name(self, name):
		try:
			sql_query = 'SELECT attempt FROM `images`.model.attempt WHERE name = $1'
			row_iter = self.cluster.query(
				sql_query,
				QueryOptions(positional_parameters=[name]))
			rows_arr = []
			for row in row_iter:
				rows_arr.append(row)
			return rows_arr[0]['attempt']
		except Exception as e:
			print(e)

	def delete_attempt_by_name(self, name):
		sql_query = 'DELETE FROM `images`.model.attempt a WHERE a.name = $1 RETURNING a.id'
		row_iter = self.cluster.query(
			sql_query,
			QueryOptions(positional_parameters=[name]))
		for row in row_iter:
			print(row)

	def get_attempt_by_id(self, id):
		try:
			sql_query = 'SELECT attempt FROM `images`.model.attempt WHERE id = $1'
			row_iter = self.cluster.query(
				sql_query,
				QueryOptions(positional_parameters=[id]))
			rows_arr = []
			for row in row_iter:
				rows_arr.append(row)
			return rows_arr[0]['attempt']
		except Exception as e:
			print(e)

	def update_attempt(self, doc: Attempt):
		try:
			key = doc.name
			result = self.cb_coll.upsert(key, doc.dict())
		except Exception as e:
			print(e)

	def get_latest_index(self):
		try:
			sql_query = 'SELECT COUNT(*) as latest_index FROM `images`.model.attempt'
			row_iter = self.cluster.query(
				sql_query,
				QueryOptions())
			for row in row_iter:
				return row['latest_index']
		except Exception as e:
			print(e)

class ImagesDataBase:
	def __init__(self):
		username = SERVER_USERNAME
		password = SERVER_PASSWORD
		bucket_name = "images"
		auth = PasswordAuthenticator(
				username,
				password
		)
		timeout_opts = ClusterTimeoutOptions(kv_timeout=timedelta(seconds=10))
		self.cluster = Cluster('couchbase://localhost', ClusterOptions(auth, timeout_options=timeout_opts))
		self.cluster.wait_until_ready(timedelta(seconds=5))
		cb = self.cluster.bucket(bucket_name)
		self.cb_coll = cb.scope("image").collection("labelled_image")

	def insert_document(self, doc: Document):
		doc = doc.dict()
		print("\nInsert CAS: ")
		try:
			key = doc["type"] + "_" + str(doc["id"])
			result = self.cb_coll.insert(key, doc)
			print(result.cas)
		except Exception as e:
			print(e)

	def get_image_by_key(self, key):
		try:
			result = self.cb_coll.get("image_{}".format(key))
			return result.value
		except Exception as e:
			print(e)

	def get_image_keys_by_classification(self, cs):
		try:
			sql_query = 'SELECT id FROM `images`.image.labelled_image WHERE classification = $1'
			row_iter = self.cluster.query(
				sql_query,
				QueryOptions(positional_parameters=[cs]))
			rows_arr = []
			for row in row_iter:
				rows_arr.append(row)
			return rows_arr
		except Exception as e:
			print(e)

	def check_if_it_exists(self, link, cs):
		try:
			sql_query = 'SELECT uri FROM `images`.image.labelled_image WHERE classification = $1 AND uri = $2'
			row_iter = self.cluster.query(
				sql_query,
				QueryOptions(positional_parameters=[cs, link]))
			for row in row_iter:
				return row
		except Exception as e:
			print(e)
	
	def get_max_image_size(self, cs):
		try:
			sql_query = 'SELECT COUNT(*) as max_items FROM `images`.image.labelled_image WHERE classification = $1'
			row_iter = self.cluster.query(
				sql_query,
				QueryOptions(positional_parameters=[cs]))
			for row in row_iter:
				return row['max_items']
		except Exception as e:
			print(e)

	def max_image_sizes(self, label_names):
		try:
			for i in range(0,len(label_names)):
				if i == 0:
					sql_query = 'SELECT COUNT(*) as max_items FROM `images`.image.labelled_image WHERE classification = $1'
				else:
					sql_query = sql_query + ' or classification = ${}'.format(i)
		
			row_iter = self.cluster.query(
				sql_query,
				QueryOptions(positional_parameters=label_names))
			for row in row_iter:
				return row['max_items']
		except Exception as e:
			print(e)

	def random_lookup_by_classification(self, cs):
		max_size = self.get_max_image_size(cs)
		random_number = random.randint(0, max_size - 1)
		try:
			sql_query = 'SELECT (SELECT im.base64 FROM `images`.image.labelled_image AS im WHERE im.classification = $1)[$2]'
			row_iter = self.cluster.query(
				sql_query,
				QueryOptions(positional_parameters=[cs, random_number]))
			for row in row_iter:
				return row
		except Exception as e:
			print(e)
	
	def delete_by_link(self, link):
		try:
			sql_query = 'DELETE attempt FROM `images`.image.labelled_image WHERE link = $1'
			row_iter = self.cluster.query(
				sql_query,
				QueryOptions(positional_parameters=[link]))
		except Exception as e:
			print(e)

class MultipleQueries(BaseModel):
	queries: List = ["american foxhound"]
	desired_chips_name: str = "dog"
	height: int = 500
	width: int = 500
	number_of_pages: int = 2
	google_domain: str = "google.com"
	api_key: str = SERPAPI_APIKEY
	limit: int = 100
	no_cache: bool = False

class QueryCreator:
	def __init__(self, multiplequery: MultipleQueries):
		self.mq = multiplequery

	def add_to_db(self):
		for query_string in self.mq.queries:
			if "height" in self.mq.dict() and "width" in self.mq.dict() and self.mq.height != None and self.mq.width != None:
				query_string = "{} imagesize:{}x{}".format(query_string, self.mq.height, self.mq.width)

			query = Query(google_domain = self.mq.google_domain, ijn=0, q=query_string, desired_chips_name = self.mq.desired_chips_name, api_key = self.mq.api_key, limit=self.mq.limit, no_cache=self.mq.no_cache)
			db = ImagesDataBase()
			serpapi = Download(query, db)
			chips = serpapi.chips_serpapi_search()
			serpapi.serpapi_search()
			serpapi.move_all_images_to_db()
			
			if self.mq.number_of_pages > 1:
				for i in range(1,self.mq.number_of_pages):
					query.ijn = i
					query.chips = chips
					db = ImagesDataBase()
					serpapi = Download(query, db)
					serpapi.serpapi_search()
					serpapi.move_all_images_to_db()

class Query(BaseModel):
	google_domain: str = "google.com"
	limit: int | None = None
	ijn: str | None = None
	q: str
	chips: str | None = None
	desired_chips_name: str | None = None
	api_key: str = SERPAPI_APIKEY
	no_cache: bool = False

class Download:
	def __init__(self, query: Query, db: ImagesDataBase):
		self.db = db
		self.query = query
		self.results = []

	def chips_serpapi_search(self):
		params = {
			"engine": "google",
			"ijn": self.query.ijn,
			"q": self.query.q,
			"google_domain": self.query.google_domain,
			"tbm": "isch",
			"chips": self.query.chips,
			"api_key": self.query.api_key,
			"no_cache": self.query.no_cache
		}

		search = GoogleSearch(params)
		results = search.get_dict()
		suggested_results = results['suggested_searches']
		chips = [x['chips'] for x in suggested_results if x['name'] == self.query.desired_chips_name]
		if chips != []:
			self.query.chips = chips[0]
			return chips[0]

	def serpapi_search(self):
		params = {
			"engine": "google",
			"ijn": self.query.ijn,
			"q": self.query.q,
			"google_domain": self.query.google_domain,
			"tbm": "isch",
			"chips": self.query.chips,
			"api_key": self.query.api_key,
			"no_cache": self.query.no_cache
		}
		self.limit = self.query.limit
		search = GoogleSearch(params)
		results = search.get_dict()
		results = results['images_results']
		self.results = results = [x['original'] for x in results]

	def get_document(self, link):
		print("Downloading {}".format(link))
		classification = self.query.q
		r = requests.get(link)
		base64_str = base64.b64encode(r.content).decode('ascii')
		extension = mimetypes.guess_extension(r.headers.get('content-type', '').split(';')[0])
		id = uuid.uuid1().hex
		if extension == ".jpg" or extension == ".jpeg" or extension == ".png":
			doc = Document(id = id, classification = classification, base64 = base64_str, uri = link )
			return doc
		else:
			return None
		
	def move_to_db(self, link):
		doc = self.get_document(link)
		sameness = self.db.check_if_it_exists(self.query.q, link)
		if doc is not None and sameness is None:
			self.db.insert_document(doc=doc)

	def move_all_images_to_db(self):
		self.serpapi_search()
		move_counter = 0
		for result in self.results:
			if move_counter == self.limit:
				break
			try: 
				self.move_to_db(result)
				move_counter = move_counter + 1
			except:
				"\n Passed image"