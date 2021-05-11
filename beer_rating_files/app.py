import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
import joblib



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
beer_scaler = pickle.load(open('scaler.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    with open('categorical_vars.json') as f:
        convert_cats = json.load(f)
    
    brewery_id = int(request.form['brewery_id'])
    review_aroma = float(request.form['review_aroma'])
    review_appearance = float(request.form['review_appearance'])
    review_palate = float(request.form['review_palate'])
    review_taste = float(request.form['review_taste'])
    beer_abv = float(request.form['beer_abv'])
    #convert categorical variables
    beer_style_dict = {value:key for key, value in convert_cats['beer_style'].items()}
    beer_style = int(beer_style_dict[str(request.form['beer_style'])])
    beer_name_dict = {value:key for key, value in convert_cats['beer_name'].items()}
    beer_name = int(beer_name_dict[str(request.form['beer_name'])])
    features = [[brewery_id, review_aroma, review_appearance, 
                review_palate,review_taste, beer_abv,
                beer_style, beer_name]]

    # features = [[int(x) for x in request.form.values()]]
    
    scale_features = beer_scaler.transform(features)
    prediction = beer_model.predict(scale_features)
    print(prediction)
    classes = np.array(["Not Excellent", "Excellent"])

    output = classes[prediction][0]

    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))
    # return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)