from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import json
import os
import csv
import base64

from dotenv import load_dotenv
from google.cloud import vision

from services.chatbot_service import ChatbotService, ChatbotServiceError

load_dotenv()

app = Flask(__name__)
app.secret_key = 'ShilpySecret123!'

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "users.json")
PREF_FILE = os.path.join(DATA_DIR, "preferences.json")
RECIPES_CSV = os.path.join(DATA_DIR, "recipes.csv")
PROGRESS_FILE = os.path.join(DATA_DIR, "progress_log.json")

chatbot_service = ChatbotService()


@app.route("/chatbot", methods=["POST"])
def chatbot():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"type": "chat", "message": "Please enter a message.", "meal_plan": []}), 400

    try:
        result = chatbot_service.handle_query(query)
        return jsonify(result)
    except ChatbotServiceError as exc:
        print("Chatbot configuration error:", exc)
        return jsonify({"type": "chat", "message": "", "meal_plan": [], "error": str(exc)}), 500
    except Exception as exc:
        print("Chatbot ERROR:", exc)
        return jsonify({"type": "chat", "message": "", "meal_plan": [], "error": str(exc)}), 500


client = vision.ImageAnnotatorClient.from_service_account_file("key.json")


def detect_food(image_base64):
    image_bytes = base64.b64decode(image_base64.split(',')[1])

    image = vision.Image(content=image_bytes)

    response = client.label_detection(image=image)
    labels = response.label_annotations

    print("FULL LABELS:", labels)

    ignore_words = ["food", "dish", "cuisine", "meal", "ingredient"]

    for label in labels:
        name = label.description.lower()
        print("Label:", name)

        if name not in ignore_words:
            return name

    return "Unknown"


@app.route("/chat")
def chat():
    return render_template("chat.html")


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
                    except Exception:
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
            if user['email'] == email and check_password_hash(user['password'], password):
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
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        user_data = {
            "name": request.form['name'],
            "email": request.form['email'],
            "password": hashed_password
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
            session['user'] = email_value
            flash("Login Successful! 🎉 Welcome to FitMeal!", "success")
            return redirect(url_for('home'))
        flash("❌ Invalid email or password. Please try again.", "danger")
        return render_template('login.html', email=email_value)
    return render_template('login.html', email=email_value)

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been successfully logged out.", "info")
    return redirect(url_for('login'))

@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    if request.method == 'POST':
        data = {
            "email": session.get('user'),
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


def calculate_cal_goal(user):
    if not user:
        return 2000
    age = int(float(user.get("age") or 25))
    height_cm = float(user.get("height") or 160)
    weight_kg = float(user.get("weight") or 56)
    gender = user.get("gender", "Female")
    activity = float(user.get("activity") or 1.2)
    goal = user.get("goal", "Maintain Weight")

    if gender.lower().startswith("m"):
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    elif gender.lower().startswith("f"):
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age

    tdee = bmr * activity
    if goal == "Lose Weight":
        cal_goal = tdee - 500
    elif goal == "Gain Weight":
        cal_goal = tdee + 500
    elif goal == "Build Muscle":
        cal_goal = tdee + 250
    else:
        cal_goal = tdee
    return int(round(cal_goal))

@app.route('/recipes')
def recipes():
    user = None
    if os.path.exists(PREF_FILE):
        with open(PREF_FILE, 'r') as f:
            prefs = json.load(f)
            if prefs:
                user_email = session.get('user')
                user_prefs = [p for p in prefs if p.get('email') == user_email]
                user = user_prefs[-1] if user_prefs else prefs[-1]
    
    cal_goal = calculate_cal_goal(user)        
    return render_template('recipes.html', user=user, cal_goal=cal_goal)


@app.route("/scan")
def scan():
    return render_template("scan.html")


@app.route('/mealplan')
def mealplan():
    return render_template('mealplan.html')


@app.route("/scan-food", methods=["POST"])
def scan_food():
    try:
        data = request.get_json()
        image = data["image"]

        food = detect_food(image)

        print("Detected:", food)

        return jsonify({
            "food": food,
            "calories": "testing"
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "food": "error",
            "calories": "0"
        })


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
        contacts_file = os.path.join(DATA_DIR, "contacts.json")
        if os.path.exists(contacts_file):
            with open(contacts_file, 'r') as f:
                contacts = json.load(f)
        else:
            contacts = []
        contacts.append(data)
        with open(contacts_file, 'w') as f:
            json.dump(contacts, f, indent=4)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": True})
        flash('Thank you for your response. Your response is submitted.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')


@app.route('/about')
def about():
    return render_template('about.html')


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
            except Exception:
                logs = []
    data['email'] = session.get('user')
    logs.append(data)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(logs, f, indent=2)
    return jsonify({"success": True})


@app.route('/report')
def report():
    if 'user' not in session:
        flash("Please log in to view your report.", "warning")
        return redirect(url_for('login'))

    user = None
    if os.path.exists(PREF_FILE):
        with open(PREF_FILE, 'r') as f:
            prefs = json.load(f)
            if prefs:
                user_email = session.get('user')
                user_prefs = [p for p in prefs if p.get('email') == user_email]
                if user_prefs:
                    user = user_prefs[-1]

    if not user:
        user = {
            "age": "25",
            "height": "160",
            "weight": "56",
            "gender": "Female",
            "waist": "34",
            "chest": "36",
            "hips": "40",
            "goal": "Lose Weight",
            "activity": "1.2",
            "diet": "Vegetarian"
        }

    age = int(float(user.get("age") or 25))
    weight_kg = float(user.get("weight") or 56)
    goal = user.get("goal", "Maintain Weight")

    cal_goal = calculate_cal_goal(user)

    if "lose" in goal.lower() or "build" in goal.lower():
        protein_goal = round(weight_kg * 1.6)
    else:
        protein_goal = round(weight_kg * 1.2)

    logs = []
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try:
                all_logs = json.load(f)
                logs = [l for l in all_logs if l.get('email') == session.get('user')]
            except Exception:
                logs = []

    dates = [entry["date"] for entry in logs if "date" in entry]
    weights = [entry["weight"] for entry in logs if "weight" in entry]
    start_weight = float(logs[0]["weight"]) if logs and "weight" in logs[0] else weight_kg
    current_weight = float(logs[-1]["weight"]) if logs and "weight" in logs[-1] else weight_kg
    latest_waist = next((entry.get("waist") for entry in reversed(logs) if entry.get("waist") not in (None, "")), user.get("waist", ""))
    latest_chest = next((entry.get("chest") for entry in reversed(logs) if entry.get("chest") not in (None, "")), user.get("chest", ""))
    last_check_in = next((entry.get("date") for entry in reversed(logs) if entry.get("date")), "")

    if "lose" in goal.lower():
        goal_weight = round(start_weight * 0.95, 1)
    elif "gain" in goal.lower() or "build" in goal.lower():
        goal_weight = round(start_weight * 1.05, 1)
    else:
        goal_weight = round(start_weight, 1)

    bmi = round(current_weight / ((height_cm / 100) ** 2), 1) if height_cm else 0
    weight_change = round(current_weight - start_weight, 1)

    progress_meta = {
        "daysFollowing": len(logs) if logs else 0,
        "startWeight": start_weight,
        "currentWeight": current_weight,
        "waist": latest_waist,
        "chest": latest_chest,
        "goal": goal,
        "goalWeight": goal_weight,
        "calorieGoal": cal_goal,
        "proteinGoal": protein_goal,
        "caloriesToday": 0,
        "proteinToday": 0,
        "lastCheckIn": last_check_in,
        "bmi": bmi,
        "weightChange": weight_change,
        "heightCm": round(height_cm),
        "activity": activity
    }

    return render_template(
        'report.html',
        progress=progress_meta,
        logs=logs,
        user=user,
        dates=dates,
        weights=weights
    )


if __name__ == '__main__':
    app.run(debug=True)
