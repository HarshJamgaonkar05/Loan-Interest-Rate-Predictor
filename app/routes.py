from flask import Blueprint, render_template, request
from .utils import predict_interest_rate
from .utils import explain_prediction  

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        form_data = request.form.to_dict()
        prediction = predict_interest_rate(form_data)
        prediction = round(prediction, 2)
    return render_template("index.html", prediction=prediction)


@main.route("/explain", methods=['POST'])
def explain():
    form_data = request.form.to_dict()
    explanation = explain_prediction(form_data)  # new function in utils
    return render_template("explain.html", explanation=explanation, form_data=form_data)
