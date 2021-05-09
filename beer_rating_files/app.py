import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle



def return_prediction(model,scaler,sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
    
    aroma= sample_json['review_aroma']
    appearance = sample_json['review_appearance']
    palate = sample_json['review_palate']
    taste = sample_json['review_taste']
    abv = sample_json['beer_abv']
    
    beer = [[aroma,appearance,palate,taste,abv]]
    
    scaled_test = scaler.transform(X_test)
    
    # classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    class_ind = model.predict_classes(X_test)
    
    return classes[X_test][0]



# REMEMBER TO LOAD THE MODEL AND THE SCALER!
app = Flask(__name__)
beer_model = pickle.load(open('logmodel.pkl', 'rb'))
beer_scaler = joblib.load("scaler.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 1)

    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)