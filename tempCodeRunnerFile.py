from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import json, os, csv

app = Flask(__name__)
app.secret_key = 'your_secret_key'

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "users.json")


# --------- API ROUTES ---------
@app.route('/api/recipes')
def api_recipes():
    recipes = []
    with open('data/FitMeal_Recipes_Updated_All_Recipes.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key in ['calories', 'protein', 'carbs', 'fat']:
                if key in row:
                    try:
                        row[key] = float(row[key]) if '.' in row[key] else int(row[key])
                    except:
                        row[key] = 0
            recipes.append(row)
    return jsonify(recipes)

# --------- SIGNUP & AUTH ---------
def save_user(data):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as file:
            users = json.load(file)
    else:
        users = []
    users.append(data)
    with open(DATA_FILE, 'w') as file:
        json.dump(users, file, indent=4)

def validate_login(email, password):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as file:
            users = json.load(file)
        for user in users:
            if user['email'] == email and user['password'] == password:
                return True
    return False

@app.route('/')
def root():
    return redirect(url_for('signup'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        user_data = {
            "name": request.form['name'],
            "email": request.form['email'],
            "password": request.form['password']
        }
        save_user(user_data)
        flash("Account created successfully! Please login to continue.", "success")
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    email_value = ""
    if request.method == 'POST':
        email_value = request.form['email']
        password = request.form['password']
        if validate_login(email_value, password):
            flash("Login Successful! 🎉 Welcome to FitMeal!", "success")
            return redirect(url_for('home'))
        flash("❌ Invalid email or password. Please try again.", "danger")
        return render_template('login.html', email=email_value)
    return render_template('login.html', email=email_value)

@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    if request.method == 'POST':
        data = {
            "age": request.form.get('age'),
            "height": request.form.get('height'),
            "weight": request.form.get('weight'),
            "gender": request.form.get('gender'),
            "waist": request.form.get('waist'),
            "chest": request.form.get('chest'),
            "hips": request.form.get('hips'),
            "diet": request.form.get('diet'),
            "goal": request.form.get('goal'),
            "activity": request.form.get('activity')
        }
        print("Received preferences:", data)
        pref_file = os.path.join(DATA_DIR, "preferences.json")
        if os.path.exists(pref_file):
            with open(pref_file, 'r') as f:
                prefs = json.load(f)
        else:
            prefs = []
        prefs.append(data)
        with open(pref_file, 'w') as f:
            json.dump(prefs, f, indent=4)
        flash('Preferences saved successfully!', 'success')
        return redirect(url_for('recipes'))
    return render_template('preferences.html')

@app.route('/recipes')
def recipes():
    # Load latest preferences
    pref_file = os.path.join(DATA_DIR, "preferences.json")
    user = None
    if os.path.exists(pref_file):
        with open(pref_file, 'r') as f:
            prefs = json.load(f)
            if prefs:
                user = prefs[-1]
    return render_template('recipes.html',user=user)

if __name__ == '__main__':
    app.run(debug=True)

