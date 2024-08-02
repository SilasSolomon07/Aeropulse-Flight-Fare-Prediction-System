from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('best_flight_fare_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Example dictionaries (populate these with your actual mappings)
airline_dict = {'Jet Airways': 0, 'IndiGo': 1, 'Air India': 2, 'SpiceJet': 3, 'Vistara': 4, 'Air Asia': 5,
                'GoAir': 6, 'Trujet': 7, 'Vistara Premium economy': 8, 'Jet Airways Business': 9,
                'Multiple carriers': 10, 'Multiple carriers Premium economy': 11}
source_dict = {'Delhi': 0, 'Kolkata': 1, 'Banglore': 2, 'Mumbai': 3, 'Chennai': 4}
destination_dict = {'Cochin': 0, 'Banglore': 1, 'Delhi': 2, 'Hyderabad': 3, 'Kolkata': 4}
stopage_dict = {'Non-stop': 0, '1 Stop': 1, '2 Stops': 2, '3 Stops': 3, "4 Stops": 4}

@app.route('/')
def home():
    return render_template('index.html',
                           airline_dict=airline_dict,
                           source_dict=source_dict,
                           destination_dict=destination_dict,
                           stopage_dict=stopage_dict)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse inputs
        airline = request.form['airline'].strip()
        source = request.form['source'].strip()
        destination = request.form['destination'].strip()
        date = request.form['date'].strip()
        stops = request.form['stops'].strip()

        # Convert inputs to model-ready format
        day, month, year = map(int, date.split('/'))

        # Validation checks
        if airline not in airline_dict:
            return render_template('index.html', prediction_text=f'Error: Airline "{airline}" not recognized.')
        if source not in source_dict:
            return render_template('index.html', prediction_text=f'Error: Source "{source}" not recognized.')
        if destination not in destination_dict:
            return render_template('index.html', prediction_text=f'Error: Destination "{destination}" not recognized.')
        if stops not in stopage_dict:
            return render_template('index.html', prediction_text=f'Error: Stoppage "{stops}" not recognized.')

        airline_num = airline_dict[airline]
        source_num = source_dict[source]
        destination_num = destination_dict[destination]
        stops_num = stopage_dict[stops]

        # Prepare input array for the model
        input_features = np.array([[airline_num, source_num, destination_num, stops_num, day, month, year]])

        # Predict fare
        predicted_fare = best_model.predict(input_features)[0]
        return render_template('index.html', prediction_text=f'Predicted Fare: Rs. {predicted_fare:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)