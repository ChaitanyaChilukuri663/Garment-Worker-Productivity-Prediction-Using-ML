from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the saved model
        model = load('random_forest_model.joblib')

        # Extract features from the request form
        quarter = float(request.form['quarter'])
        department = float(request.form['department'])
        day = float(request.form['day'])
        team = float(request.form['team'])
        targeted_productivity = float(request.form['targeted_productivity'])
        standard_minute_value = float(request.form['standard_minute_value'])
        work_in_progress = float(request.form['work_in_progress'])
        over_time = float(request.form['over_time'])
        incentive = float(request.form['incentive'])
        idle_men = float(request.form['idle_men'])
        no_of_style_change = float(request.form['no_of_style_change'])
        no_of_workers = float(request.form['no_of_workers'])

        # Make a prediction
        prediction = model.predict([[quarter, department, day, team, targeted_productivity,
                                     standard_minute_value, work_in_progress, over_time,
                                     incentive, idle_men, no_of_style_change, no_of_workers]])

        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        print(f"Error: {e}")
        return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)
