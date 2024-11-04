from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
# import mlflow.pyfunc
import mlflow
import pandas as pd
app = FastAPI()


# def fetch_latest_model():
#     client = MlflowClient()
#     return dict(client.search_registered_models()[0])["name"]


# def fetch_latest_version(model_name):
#     model = mlflow.pyfunc.load_model(
#         model_uri=f"models:/{model_name}/Production"
#     )
#     return model

from mlflow import MlflowClient
client = MlflowClient()
def fetch_best_model():
    # fetch best run experiment
    exp_id = client.search_experiments()[0].experiment_id
    best_run = client.search_runs(exp_id ,order_by=["metrics.training_score DESC"], max_results=1)
    #Fetching Run ID for
    run_id = best_run[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    print(model_uri)
    model = mlflow.sklearn.load_model(model_uri=model_uri)
    return model


instrumentator = Instrumentator().instrument(app)
@app.on_event("startup")
async def _startup():
    instrumentator.expose(app)

@app.get("/predict/")
def model_output(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    print("Works I")
    model = fetch_best_model()
    print("Works II")
    input = pd.DataFrame({"sepal_length": [sepal_length], "sepal_width": [sepal_width], "petal_length": [petal_length], "petal_width": [petal_width]})

    prediction = model.predict(input)
    print(prediction)
    return {"prediction": prediction[0]}
