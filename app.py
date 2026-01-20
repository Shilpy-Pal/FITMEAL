from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import json, os, csv
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "users.json")
PREF_FILE = os.path.join(DATA_DIR, "preferences.json")
RECIPES_CSV = os.path.join(DATA_DIR, "recipes.csv")
PROGRESS_FILE = os.path.join(DATA_DIR, "progress_log.json")


# --------- API ROUTES ---------

@app.route('/api/recipes')
def api_recipes():
    recipes = []
    with open(RECIPES_CSV, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key in ['calories', 'protein', 'carbs', 'fat']:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key]) if '.' in row[key] else int(row[key])
                    except:
                        row[key] = 0
                else:
                    row[key] = 0

            img = row.get("image", "")
            if img and img.startswith("staticimages"):
                fname = img.replace("staticimages", "").lstrip("/\\")
                img = f"/static/images/{fname}"
            row["image"] = img

            ing_raw = row.get("ingredients", "") or ""
            if "|" in ing_raw:
                ing_list = [i.strip() for i in ing_raw.split("|") if i.strip()]
            else:
                ing_list = [i.strip() for i in ing_raw.split(",") if i.strip()] if ing_raw else []

            instr_raw = row.get("instructions", "") or ""
            if "|" in instr_raw:
                instr_list = [s.strip() for s in instr_raw.split("|") if s.strip()]
            else:
                instr_list = [s.strip() for s in instr_raw.split(". ") if s.strip()] if instr_raw else []

            recipes.append({
                "name": row.get("name", ""),
                "calories": row.get("calories", 0),
                "dietary": row.get("dietary", ""),
                "type": row.get("type", ""),
                "portion": row.get("portion", ""),
                "image": row.get("image", ""),
                "prep_time": row.get("prep_time", ""),
                "ingredients": ing_list,
                "instructions": instr_list,
                "protein": row.get("protein", 0),
                "carbs": row.get("carbs", 0),
                "fat": row.get("fat", 0)
            })

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


# --------- PREFERENCES / RECIPES / MEALPLAN ---------

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
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        if os.path.exists(PREF_FILE):
            with open(PREF_FILE, 'r') as f:
                prefs = json.load(f)
        else:
            prefs = []
        prefs.append(data)
        with open(PREF_FILE, 'w') as f:
            json.dump(prefs, f, indent=4)
        flash('Preferences saved successfully!', 'success')
        return redirect(url_for('recipes'))
    return render_template('preferences.html')

@app.route('/recipes')
def recipes():
    user = None
    if os.path.exists(PREF_FILE):
        with open(PREF_FILE, 'r') as f:
            prefs = json.load(f)
            if prefs:
                user = prefs[-1]
    return render_template('recipes.html', user=user)

@app.route('/mealplan')
def mealplan():
    return render_template('mealplan.html')


# --------- CONTACT / ABOUT ---------

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        data = {
            "name": request.form['name'],
            "email": request.form['email'],
            "message": request.form['message']
        }
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        CONTACTS_FILE = os.path.join(DATA_DIR, "contacts.json")
        if os.path.exists(CONTACTS_FILE):
            with open(CONTACTS_FILE, 'r') as f:
                contacts = json.load(f)
        else:
            contacts = []
        contacts.append(data)
        with open(CONTACTS_FILE, 'w') as f:
            json.dump(contacts, f, indent=4)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": True})
        flash('Thank you for your response. Your response is submitted.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')


# --------- PROGRESS LOG API ---------

@app.route('/api/save_progress', methods=['POST'])
def save_progress():
    data = request.get_json() or {}
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    logs = []
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    logs.append(data)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(logs, f, indent=2)
    return jsonify({"success": True})


# --------- REPORT PAGE ---------

@app.route('/report')
def report():
    user = None
    if os.path.exists(PREF_FILE):
        with open(PREF_FILE, 'r') as f:
            prefs = json.load(f)
            if prefs:
                user = prefs[-1]

    if not user:
        user = {
            "age": "25",
            "height": "160",
            "weight": "56",
            "gender": "Female",
            "waist": "34",
            "chest": "36",
            "hips": "40",
            "diet": "Vegetarian",
            "goal": "Lose Weight",
            "activity": "1.2"
        }

    age = int(float(user.get("age") or 25))
    height_cm = float(user.get("height") or 160)
    weight_kg = float(user.get("weight") or 56)
    gender = user.get("gender", "Female")
    activity = float(user.get("activity") or 1.2)
    goal = user.get("goal", "Maintain Weight")

    # Calculate BMR
    if gender.lower().startswith("m"):
        bmr = 88.362 + 13.397 * weight_kg + 4.799 * height_cm - 5.677 * age
    else:
        bmr = 447.593 + 9.247 * weight_kg + 3.098 * height_cm - 4.330 * age

    # Calculate TDEE (Total Daily Energy Expenditure)
    tdee = bmr * activity
    if "lose" in goal.lower():
        cal_goal = tdee - 300
    elif "gain" in goal.lower():
        cal_goal = tdee + 250
    else:
        cal_goal = tdee
    cal_goal = int(round(cal_goal))

    # Protein goal based on goal
    if "lose" in goal.lower() or "build" in goal.lower():
        protein_goal = round(weight_kg * 1.6)
    else:
        protein_goal = round(weight_kg * 1.2)

    # Load progress logs
    logs = []
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try:
                logs = json.load(f)
            except:
                logs = []

    # Extract dates and weights for chart
    dates = [entry["date"] for entry in logs if "date" in entry]
    weights = [entry["weight"] for entry in logs if "weight" in entry]

    progress_meta = {
        "daysFollowing": len(logs) if logs else 0,
        "startWeight": logs[0]["weight"] if logs else weight_kg,
        "currentWeight": logs[-1]["weight"] if logs else weight_kg,
        "waist": user.get("waist", ""),
        "chest": user.get("chest", ""),
        "goal": goal,
        "calorieGoal": cal_goal,
        "proteinGoal": protein_goal,
        "caloriesToday": 0,
        "proteinToday": 0
    }

    return render_template('report.html',
                           progress=progress_meta,
                           logs=logs,
                           user=user,
                           dates=dates,
                           weights=weights)

if __name__ == '__main__':
    app.run(debug=True)
