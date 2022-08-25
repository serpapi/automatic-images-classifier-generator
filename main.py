from fastapi import FastAPI, BackgroundTasks
from add_couchbase import *
from dataset import CustomImageDataLoader, CustomImageDataset
from train import Train
from models import *
from validationtest import ValidationTest
from commands import *
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()
templates = Jinja2Templates(directory="templates/")
partials = Jinja2Templates(directory="static/partials/")

app.mount(
	"/static",
	StaticFiles(directory=Path(__file__).parent.absolute() / "static"),
	name="static",
)

@app.get("/")
def read_root():
	return {"Hello": "World"}

@app.post("/add_to_db/")
async def create_query(query: Query, background_tasks: BackgroundTasks):
	db = ImagesDataBase()
	serpapi = Download(query, db)
	def background_move(serpapi):
		serpapi.move_all_images_to_db()
	
	background_tasks.add_task(background_move, serpapi)
	return  {"status": "Complete"}

@app.post("/multiple_query/")
async def create_multiple_query(multiplequery: MultipleQueries, background_tasks: BackgroundTasks):
	serpapi = QueryCreator(multiplequery)
	def background_multiple_query(serpapi):
		serpapi.add_to_db()
	
	background_tasks.add_task(background_multiple_query, serpapi)
	return  {"status": "Complete"}

@app.post("/train/")
async def train(tc: TrainCommands, background_tasks: BackgroundTasks):
	def background_training(tc):
		if 'name' in tc.model and tc.model['name'] != "":
			model = eval(tc.model['name'])
		else:
			model = CustomModel

		try:
			a = find_attempt(name = tc.model_name)
			a["status"] = "Training"
			a["training_losses"] = []
			a = Attempt(**a)
			update_attempt(a)
			index = a.id
		except:
			index = return_index()['status']
			a = Attempt(name=tc.model_name, training_commands = tc.dict(), status = "Training", n_epoch=tc.n_epoch, id=index)
			create_attempt(a=a)

		trainer = Train(tc, model, CustomImageDataLoader, CustomImageDataset, ImagesDataBase)
		trainer.train()
		model = None
		try:
			torch.cuda.empty_cache()
		except:
			pass

		try:
			a = find_attempt(name = tc.model_name)
			a["status"] = "Trained"
			a = Attempt(**a)
			update_attempt(a)
		except:
			return {"status": "Incomplete"}
	
	background_tasks.add_task(background_training, tc)
	return {"status": "Complete"}

@app.post("/test/")
async def validationtest(tc: ValidationTestCommands, background_tasks: BackgroundTasks):
	def background_testing(tc):
		if 'name' in tc.model and tc.model['name'] != "":
			model = eval(tc.model['name'])
		else:
			model = CustomModel

		try:
			a = find_attempt(name = tc.model_name)
			a["testing_commands"] = tc.dict()
			a["status"] = "Testing"
			a = Attempt(**a)
			update_attempt(a)
		except:
			return {"status": "No Model Attempt by that Name"}

		tester = ValidationTest(tc, CustomImageDataset, ImagesDataBase, model)
		accuracy = tester.test_accuracy()
		
		a = find_attempt(name = tc.model_name)
		a["accuracy"] = accuracy
		a["status"] = "Complete"
		a = Attempt(**a)
		update_attempt(a)
		
		model = None
		try:
			torch.cuda.empty_cache()
		except:
			pass
	
	background_tasks.add_task(background_testing, tc)
	return {"status": "Success"}

@app.post("/create_attempt")
def create_attempt(a: Attempt):
	db = ModelsDatabase()
	db.insert_attempt(a)
	return {"status": "Success"}

@app.post("/find_attempt/")
def find_attempt(name: str):
	db = ModelsDatabase()
	attempt = db.get_attempt_by_name(name)
	return attempt

@app.post("/update_attempt")
def update_attempt(a: Attempt):
	db = ModelsDatabase()
	db.update_attempt(a)
	return {"status": "Success"}

@app.post("/latest_attempt_index")
def return_index():
	db = ModelsDatabase()
	index = db.get_latest_index()
	return {"status": index}

@app.post("/delete_attempt/")
def delete_attempt(name: str):
	db = ModelsDatabase()
	db.delete_attempt_by_name(name)
	return {"status": "Success"}