import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
from marshmallow import Schema, fields, INCLUDE


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

class RequestSchema(Schema):
    #experience = fields.Float(required=True)
    #test_score = fields.Float(required=True)
    #interview_score = fields.Float(required=True)
    # Incase we want to allow missing fields and impute them with 0
    experience = fields.Float(default=0)
    test_score = fields.Float(default=0)
    interview_score = fields.Float(default=0)

    class Meta: # This would allow unknown fields in the request body
        unknown = INCLUDE

expected_features = ["experience", "test_score", "interview_score"]
#@app.route('/')
#def home():
#    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    request_body = request.get_json()
    validation_error = RequestSchema().validate(request_body)
    response = {}
    if validation_error=={}:
        features = [float(request_body.get(x, 0)) for x in expected_features]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = round(prediction[0])
        response["salary"] = output
        # return render_template('index.html', prediction_text='Fixed Per Month Employee Salary should be Rs {} /-'.format(output))
        return response
    else:
        response["Error"] = validation_error
        return response


if __name__ == "__main__":
    app.run(host ='0.0.0.0', port = 5000, debug=True)
