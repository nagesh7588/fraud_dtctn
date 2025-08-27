from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd

app = Flask(__name__)


# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
# Load feature names
with open('features.pkl', 'rb') as f:
    FEATURES = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    for col in FEATURES:
        if col not in df:
            df[col] = 0
    df = df[FEATURES]
    pred = model.predict(df)[0]
    return jsonify({'is_fraud': int(pred)})


# Simple HTML template for UI
HTML_FORM = '''
<html>
<head><title>Fraud Detection UI</title></head>
<body>
    <h2>BFSI Fraud Detection</h2>
    <form method="post">
        <label>Transaction Amount:</label>
        <input type="number" name="transaction_amount" required><br><br>
        <label>Account Age (years):</label>
        <input type="number" name="account_age" required><br><br>
        <label>Transaction Type:</label>
        <select name="transaction_type">
            <option value="transfer">Transfer</option>
            <option value="payment">Payment</option>
            <option value="withdrawal">Withdrawal</option>
        </select><br><br>
        <label>Location:</label>
        <select name="location">
            <option value="NY">NY</option>
            <option value="CA">CA</option>
            <option value="TX">TX</option>
        </select><br><br>
        <input type="submit" value="Check Fraud">
    </form>
    {% if result is not none %}
        <h3>Prediction: {{ result }}</h3>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    error = None
    if request.method == 'POST':
        try:
            data = {
                'transaction_amount': int(request.form['transaction_amount']),
                'account_age': int(request.form['account_age']),
                'transaction_type': request.form['transaction_type'],
                'location': request.form['location']
            }
            df = pd.DataFrame([data])
            df = pd.get_dummies(df)
            # Ensure all expected columns exist and are in correct order
            for col in FEATURES:
                if col not in df.columns:
                    df[col] = 0
            df = df[FEATURES]
            # Remove any extra columns
            df = df.loc[:, FEATURES]
            pred = model.predict(df)[0]
            result = 'Fraudulent' if pred == 1 else 'Not Fraudulent'
        except Exception as e:
            error = f"Error: {str(e)}"
    return render_template_string(HTML_FORM + ("<p style='color:red;'>" + error + "</p>" if error else ""), result=result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
