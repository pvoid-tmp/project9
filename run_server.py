import dill
import pandas as pd
import flask
import os

dill._dill._reverse_typemap['ClassType'] = type

app = flask.Flask(__name__)
model = None


def load_model(model_path):
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)


@app.route("/", methods=["GET"])
def general():
	return "Welcome to prediction process"


@app.route("/predict", methods=["POST"])
def predict():
	data = {"success": False}
	if flask.request.method == "POST":
		text = ""
		request_json = flask.request.get_json()
		if request_json["text"]:
			text = request_json['text']
		preds = model.predict_proba(pd.DataFrame({"text": [text]}))
		data["predictions"] = preds[:, 1][0]
		data["text"] = text
		data["success"] = True
	return flask.jsonify(data)


if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	modelpath = "app/app/models/logreg_pipeline.dill"
	load_model(modelpath)
#	print(model)
#	app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 8180)))
	app.run()
