import os
import re
import json
import requests
from typing import Any
from html import unescape
class ChatbotServiceError(Exception):
    """Raised when the chatbot service cannot be initialized."""


class ChatbotService:
    HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
    SPOONACULAR_URL = "https://api.spoonacular.com/recipes/complexSearch"
    DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

    def __init__(self) -> None:
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY", "").strip()
        self.spoonacular_api_key = os.getenv("SPOONACULAR_API_KEY", "").strip()
        self.hf_model = os.getenv("HUGGINGFACE_MODEL", self.DEFAULT_MODEL).strip()
        self.hf_api_url = os.getenv("HUGGINGFACE_API_URL", self.HF_API_URL).strip()
        self.timeout = int(os.getenv("CHATBOT_HTTP_TIMEOUT", "45"))
        self.session = requests.Session()

    def handle_query(self, query: str) -> dict[str, Any]:
        normalized_query = query.strip()
        if not normalized_query:
            return {"type": "chat", "message": "Please type a message so I can help.", "meal_plan": []}

        classification = self._classify_query(normalized_query)

        if classification != "food":
            return {
                "type": "chat",
                "message": self._generate_chat_reply(normalized_query),
                "meal_plan": [],
            }

        meal_plan = self.build_meal_plan(normalized_query)
        return {
            "type": "meal_plan",
            "message": "Your meal plan is ready. Preview it below or download it.",
            "meal_plan": meal_plan,
        }

    def build_meal_plan(self, query: str) -> list[dict[str, Any]]:
        if not self.hf_api_key:
            raise ChatbotServiceError("Missing HUGGINGFACE_API_KEY environment variable.")

        try:
            preferences = self._extract_preferences(query)
        except Exception:
            preferences = self._basic_preferences(query)

        meals = int(preferences.get("meals") or 3)
        meals = max(1, min(meals, 5))
        preferences["meals"] = meals

        explicit_calories = self._extract_explicit_calories(query)
        if explicit_calories is not None:
            preferences["calories"] = explicit_calories
            preferences["calorie_range"] = f"{max(explicit_calories - 100, 0)}-{explicit_calories + 100}"

        if not preferences.get("calories"):
            inferred_calories = preferences.get("target_calories")
            if isinstance(inferred_calories, (int, float)):
                preferences["calories"] = int(inferred_calories)
            else:
                preferences["calories"] = self._heuristic_calories(query, meals)

        if not preferences.get("diet_type"):
            preferences["diet_type"] = self._extract_diet_from_text(query)

        try:
            meal_plan = self._fetch_spoonacular_meals(preferences)
        except Exception:
            meal_plan = []

        if not meal_plan:
            try:
                meal_plan = self._generate_fallback_meals(query, preferences)
            except Exception:
                meal_plan = self._basic_fallback(preferences)
        elif len(meal_plan) < meals:
            try:
                fallback_meals = self._generate_fallback_meals(query, preferences)
            except Exception:
                fallback_meals = self._basic_fallback(preferences)
            meal_plan.extend(fallback_meals[len(meal_plan):meals])

        return meal_plan[:meals]

    def _basic_preferences(self, query: str) -> dict[str, Any]:
        meals = self._safe_int(re.search(r"(\d)\s*(?:meals|meal)", query, re.IGNORECASE).group(1)) if re.search(r"(\d)\s*(?:meals|meal)", query, re.IGNORECASE) else 3
        explicit_calories = self._extract_explicit_calories(query)
        target_calories = explicit_calories or self._heuristic_calories(query, meals)
        return {
            "calories": explicit_calories,
            "calorie_range": f"{max(target_calories - 150, 0)}-{target_calories + 150}",
            "target_calories": target_calories,
            "diet_type": self._extract_diet_from_text(query),
            "meals": meals,
            "search_terms": query.strip(),
        }

    def _extract_preferences(self, query: str) -> dict[str, Any]:
        prompt = (
            "You are a nutrition extraction assistant. "
            "Understand Hinglish, English, Hindi transliteration, and spelling mistakes. "
            "Read the user request and return only valid JSON with these keys: "
            "calories, calorie_range, target_calories, diet_type, meals, search_terms. "
            "Rules: "
            "1. If user explicitly gives calories, set calories to that exact number. "
            "2. If calories are not given, infer a realistic daily calorie target based on the request and set target_calories plus calorie_range. "
            "3. diet_type must be vegetarian, non-vegetarian, or any. "
            "4. meals must be an integer from 1 to 5. "
            "5. search_terms should be a short search query for recipes. "
            "6. No markdown, no explanation, JSON only.\n"
            f"User query: {query}"
        )

        data = self._call_huggingface_json(prompt)
        return {
            "calories": self._safe_int(data.get("calories")),
            "calorie_range": data.get("calorie_range"),
            "target_calories": self._safe_int(data.get("target_calories")),
            "diet_type": self._normalize_diet(data.get("diet_type")),
            "meals": self._safe_int(data.get("meals")) or 3,
            "search_terms": (data.get("search_terms") or query).strip(),
        }

    def _fetch_spoonacular_meals(self, preferences: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.spoonacular_api_key:
            return []

        meals = preferences["meals"]
        target_daily_calories = max(int(preferences["calories"]), 1)
        per_meal_target = max(round(target_daily_calories / meals), 150)

        params: dict[str, Any] = {
            "apiKey": self.spoonacular_api_key,
            "query": preferences["search_terms"],
            "number": 5,
            "addRecipeInformation": "true",
            "addRecipeNutrition": "true",
            "minCalories": max(per_meal_target - 125, 50),
            "maxCalories": per_meal_target + 175,
        }

        if preferences["diet_type"] == "vegetarian":
            params["diet"] = "vegetarian"

        response = self.session.get(self.SPOONACULAR_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        results = data.get("results") or []
        if not results:
            return []

        meal_labels = self._meal_labels(meals)
        meal_plan: list[dict[str, Any]] = []

        for index, recipe in enumerate(results[:meals]):
            nutrients = self._extract_nutrients(recipe.get("nutrition", {}).get("nutrients", []))
            ingredients = [
                ingredient.get("original") or ingredient.get("originalName") or ingredient.get("name", "")
                for ingredient in recipe.get("extendedIngredients", [])
                if ingredient.get("original") or ingredient.get("originalName") or ingredient.get("name")
            ]

            instructions = self._format_instructions(recipe)

            meal_plan.append({
                "meal": meal_labels[index],
                "title": recipe.get("title", ""),
                "image": recipe.get("image") or recipe.get("imageUrl") or "",
                "calories": nutrients.get("Calories", ""),
                "protein": nutrients.get("Protein", ""),
                "carbs": nutrients.get("Carbohydrates", ""),
                "fat": nutrients.get("Fat", ""),
                "ingredients": ingredients,
                "instructions": instructions,
            })

        return meal_plan

    def _generate_fallback_meals(self, query: str, preferences: dict[str, Any]) -> list[dict[str, Any]]:
        meals = preferences["meals"]
        prompt = (
            "You are a meal planning assistant. "
            "Generate a practical meal plan in JSON only. "
            "Understand Hinglish and spelling mistakes. "
            "Return JSON with one key: meal_plan. "
            "meal_plan must be an array of exactly "
            f"{meals} objects. Each object must have keys: meal, title, image, calories, protein, carbs, fat, ingredients, instructions. "
            "image should be an empty string. "
            "Use approximate but realistic nutrition values. "
            f"Target daily calories: {preferences['calories']}. "
            f"Diet type: {preferences['diet_type']}. "
            f"User request: {query}"
        )

        data = self._call_huggingface_json(prompt)
        meal_plan = data.get("meal_plan")
        if isinstance(meal_plan, list) and meal_plan:
            labels = self._meal_labels(meals)
            normalized: list[dict[str, Any]] = []
            for index, item in enumerate(meal_plan[:meals]):
                normalized.append({
                    "meal": item.get("meal") or labels[index],
                    "title": item.get("title", ""),
                    "image": item.get("image", ""),
                    "calories": str(item.get("calories", "")),
                    "protein": str(item.get("protein", "")),
                    "carbs": str(item.get("carbs", "")),
                    "fat": str(item.get("fat", "")),
                    "ingredients": item.get("ingredients") if isinstance(item.get("ingredients"), list) else [],
                    "instructions": item.get("instructions", ""),
                })
            if normalized:
                return normalized

        return self._basic_fallback(preferences)

    def _call_huggingface_json(self, prompt: str) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.hf_model,
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        response = self.session.post(self.hf_api_url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return self._extract_json_object(content)

    def _call_huggingface_text(self, prompt: str, system_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.hf_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 180,
        }

        response = self.session.post(self.hf_api_url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _extract_json_object(self, content: str) -> dict[str, Any]:
        content = content.strip()
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
        if fenced_match:
            content = fenced_match.group(1)

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return {}

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _extract_nutrients(self, nutrients: list[dict[str, Any]]) -> dict[str, str]:
        wanted = {"Calories", "Protein", "Carbohydrates", "Fat"}
        values = {key: "" for key in wanted}

        for nutrient in nutrients:
            name = nutrient.get("name")
            if name not in wanted:
                continue
            amount = nutrient.get("amount")
            unit = nutrient.get("unit", "")
            if amount is None:
                continue
            values[name] = f"{round(float(amount), 1)} {unit}".strip()

        return values

    def _format_instructions(self, recipe: dict[str, Any]) -> str:
        analyzed = recipe.get("analyzedInstructions") or []
        steps: list[str] = []
        for instruction in analyzed:
            for step in instruction.get("steps", []):
                text = step.get("step", "").strip()
                if text:
                    steps.append(text)

        if steps:
            return " ".join(steps)

        raw_instructions = recipe.get("instructions") or recipe.get("summary") or ""
        return self._clean_text(raw_instructions)

    def _extract_explicit_calories(self, query: str) -> int | None:
        match = re.search(r"(\d{3,4})\s*(?:kcal|calories|calorie|cal)", query, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _extract_diet_from_text(self, query: str) -> str:
        text = query.lower()
        vegetarian_terms = ["veg", "vegetarian", "shakahari", "satvik", "paneer", "dal"]
        non_veg_terms = ["non veg", "non-veg", "chicken", "egg", "fish", "mutton"]

        if any(term in text for term in non_veg_terms):
            return "non-vegetarian"
        if any(term in text for term in vegetarian_terms):
            return "vegetarian"
        return "any"

    def _normalize_diet(self, diet: Any) -> str:
        if not isinstance(diet, str):
            return "any"

        text = diet.strip().lower()
        if text in {"veg", "vegetarian"}:
            return "vegetarian"
        if text in {"non veg", "non-veg", "non vegetarian", "non-vegetarian"}:
            return "non-vegetarian"
        return "any"

    def _heuristic_calories(self, query: str, meals: int) -> int:
        text = query.lower()
        if any(word in text for word in ["weight loss", "lose", "fat loss", "cut"]):
            return 1500 if meals <= 3 else 1600
        if any(word in text for word in ["gain", "bulk", "muscle", "build"]):
            return 2300 if meals <= 3 else 2500
        if any(word in text for word in ["high protein", "fitness", "healthy"]):
            return 1900
        return 1800

    def _meal_labels(self, meals: int) -> list[str]:
        variants = {
            1: ["Meal 1"],
            2: ["Breakfast", "Dinner"],
            3: ["Breakfast", "Lunch", "Dinner"],
            4: ["Breakfast", "Lunch", "Snack", "Dinner"],
            5: ["Breakfast", "Morning Snack", "Lunch", "Evening Snack", "Dinner"],
        }
        return variants.get(meals, [f"Meal {index + 1}" for index in range(meals)])

    def _classify_query(self, query: str) -> str:
        if not self.hf_api_key:
            return "food" if self._fallback_food_detection(query) else "general"

        prompt = (
            "Classify the user's message into exactly one label: "
            "food or general. "
            "Choose food only if the user is asking about diet, meal plans, recipes, calories, nutrition, what to eat, vegetarian or non-vegetarian food choices, or similar food planning topics. "
            "Choose general for greetings, casual chat, coding, study, facts, jokes, life advice, and everything else. "
            "Return only valid JSON in this format: {\"category\":\"food\"} or {\"category\":\"general\"}. "
            "No markdown, no explanation.\n"
            f"User query: {query}"
        )

        try:
            data = self._call_huggingface_json(prompt)
            category = str(data.get("category", "")).strip().lower()
            if category in {"food", "general"}:
                return category
        except Exception:
            pass

        return "food" if self._fallback_food_detection(query) else "general"

    def _fallback_food_detection(self, query: str) -> bool:
        text = query.lower().strip()
        if not text:
            return False

        greeting_patterns = {
            "hi", "hii", "hiii", "hello", "hey", "heyy", "heyy", "yo",
            "namaste", "hy", "hola", "good morning", "good evening", "good afternoon",
            "thanks", "thank you", "ok", "okay", "bye"
        }
        if text in greeting_patterns:
            return False

        conversational_phrases = [
            "how are you", "who are you", "what can you do", "help me",
            "can you help", "tell me about yourself"
        ]
        if any(phrase in text for phrase in conversational_phrases):
            return False

        strong_food_phrases = [
            "meal plan", "diet plan", "make me a diet", "make me a meal plan",
            "weight loss diet", "weight gain diet", "vegetarian meal plan",
            "non veg meal plan", "recipe for", "recipes for", "calorie diet",
            "high protein meal", "what should i eat", "what can i eat",
            "breakfast ideas", "lunch ideas", "dinner ideas", "snack ideas"
        ]
        if any(phrase in text for phrase in strong_food_phrases):
            return True

        food_keywords = [
            "diet", "recipe", "recipes", "calorie", "calories", "protein", "carbs", "fat",
            "vegetarian", "non veg", "non-veg", "breakfast", "lunch", "dinner", "snack",
            "food", "foods", "meal", "nutrition", "paneer", "chicken", "egg", "dal"
        ]
        food_keyword_hits = sum(1 for keyword in food_keywords if keyword in text)
        number_hits = len(re.findall(r"\b\d+\b", text))

        return food_keyword_hits >= 2 or (food_keyword_hits >= 1 and number_hits >= 1)

    def _generate_chat_reply(self, query: str) -> str:
        if not self.hf_api_key:
            return "Hi! I can chat with you normally and also help with meal plans whenever you want."

        prompt = (
            "Reply naturally to the user's message. "
            "You are a friendly general-purpose AI assistant inside a Flask web app. "
            "You can answer everyday questions, explain concepts, chat casually, and help with food or meal planning when asked. "
            "For this message, do not generate a meal plan unless the user explicitly asks for one. "
            "Keep the tone helpful, human, and conversational. "
            "You may answer in English or Hinglish depending on the user's style.\n"
            f"User: {query}"
        )
        try:
            return self._call_huggingface_text(
                prompt,
                "You are a friendly, general AI assistant. Reply in plain text only. Be natural and useful.",
            )
        except Exception:
            return "Main aapke general questions ka bhi answer de sakta hoon, aur agar chaho to meal plan bhi bana sakta hoon."

    def _basic_fallback(self, preferences: dict[str, Any]) -> list[dict[str, Any]]:
        labels = self._meal_labels(preferences["meals"])
        per_meal = max(round(preferences["calories"] / preferences["meals"]), 150)
        titles = {
            "vegetarian": [
                "Oats Upma Bowl",
                "Dal Rice Plate",
                "Paneer Wrap",
                "Sprouts Chaat",
                "Vegetable Khichdi",
            ],
            "non-vegetarian": [
                "Egg Bhurji Toast",
                "Chicken Rice Bowl",
                "Grilled Fish Plate",
                "Greek Yogurt Snack",
                "Chicken Salad Wrap",
            ],
            "any": [
                "Protein Breakfast Bowl",
                "Balanced Lunch Plate",
                "Nutritious Wrap",
                "Fruit and Yogurt Snack",
                "Simple Dinner Bowl",
            ],
        }
        picked = titles.get(preferences["diet_type"], titles["any"])

        meal_plan: list[dict[str, Any]] = []
        for index, label in enumerate(labels):
            meal_plan.append({
                "meal": label,
                "title": picked[index % len(picked)],
                "image": "",
                "calories": str(per_meal),
                "protein": "18 g",
                "carbs": "35 g",
                "fat": "12 g",
                "ingredients": ["Protein source", "Whole grains", "Vegetables", "Healthy fat"],
                "instructions": "Combine fresh ingredients, cook until done, and adjust seasoning to taste.",
            })
        return meal_plan

    def _safe_int(self, value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _clean_text(self, value: str) -> str:
        if not value:
            return ""
        text = re.sub(r"<[^>]+>", " ", unescape(value))
        text = re.sub(r"\s+", " ", text).strip()
        return text
