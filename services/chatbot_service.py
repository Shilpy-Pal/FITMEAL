import os
import re
import json
import requests
from typing import Any
from html import unescape
from datetime import datetime


class ChatbotServiceError(Exception):
    """Raised when the chatbot service cannot be initialized."""


class ChatbotService:
    HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
    SPOONACULAR_URL = "https://api.spoonacular.com/recipes/complexSearch"
    DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    FALLBACK_CHAT_MODELS = (
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
    )

    def __init__(self) -> None:
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY", "").strip()
        self.spoonacular_api_key = os.getenv("SPOONACULAR_API_KEY", "").strip()
        self.hf_model = os.getenv("HUGGINGFACE_MODEL", self.DEFAULT_MODEL).strip()
        self.hf_api_url = os.getenv("HUGGINGFACE_API_URL", self.HF_API_URL).strip()
        self.timeout = int(os.getenv("CHATBOT_HTTP_TIMEOUT", "45"))
        self.session = requests.Session()
        self.model_candidates = self._build_model_candidates()

    def handle_query(self, query: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        normalized_query = query.strip()
        if not normalized_query:
            return {"type": "chat", "message": "Please type a message so I can help.", "meal_plan": []}

        chat_history = self._sanitize_history(history)
        classification = self._classify_query(normalized_query, chat_history)
        if classification != "food" and self._fallback_food_detection(normalized_query):
            classification = "food"

        if classification == "food":
            if self._is_meal_plan_request(normalized_query, chat_history):
                meal_plan, preferences = self.build_meal_plan(normalized_query)
                report_snapshot = self._build_report_snapshot(normalized_query, meal_plan, preferences)
                return {
                    "type": "meal_plan",
                    "message": "Your meal plan is ready. Preview it below or download your dietician-style report.",
                    "meal_plan": meal_plan,
                    "report": report_snapshot,
                }

            return {
                "type": "chat",
                "message": self._generate_dietician_reply(normalized_query, chat_history),
                "meal_plan": [],
            }

        return {
            "type": "chat",
            "message": self._generate_chat_reply(normalized_query, chat_history),
            "meal_plan": [],
        }

    def build_meal_plan(self, query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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

        if self._is_weekly_plan_request(query):
            preferences["plan_scope"] = "weekly"
            preferences["meals"] = 7
            try:
                weekly_plan = self._generate_weekly_meals(query, preferences)
            except Exception:
                weekly_plan = self._weekly_template(preferences)
            return weekly_plan, preferences

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

        final_plan = meal_plan[:meals]
        return final_plan, preferences

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
        if not self.hf_api_key:
            return self._basic_preferences(query)

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
        if not self.hf_api_key:
            return self._basic_fallback(preferences)

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

    def _generate_weekly_meals(self, query: str, preferences: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.hf_api_key:
            return self._weekly_template(preferences)

        target_calories = int(preferences.get("calories") or self._heuristic_calories(query, 3))
        prompt = (
            "You are a dietician meal planner. "
            "Create a 7-day practical meal plan and return JSON only. "
            "Return object with key: meal_plan. "
            "meal_plan must be exactly 7 objects with keys: meal, title, image, calories, protein, carbs, fat, ingredients, instructions. "
            "Set meal to Day 1, Day 2 ... Day 7. "
            "title should summarize that day's menu. "
            "ingredients should be an array of key foods. "
            "instructions should contain brief guidance for the day. "
            "image should be empty string. "
            f"Diet type: {preferences.get('diet_type', 'any')}. "
            f"Target daily calories: {target_calories}. "
            f"User request: {query}"
        )

        data = self._call_huggingface_json(prompt)
        meal_plan = data.get("meal_plan")
        if not isinstance(meal_plan, list) or not meal_plan:
            return self._weekly_template(preferences)

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(meal_plan[:7]):
            normalized.append({
                "meal": item.get("meal") or f"Day {index + 1}",
                "title": item.get("title", f"Day {index + 1} plan"),
                "image": item.get("image", ""),
                "calories": str(item.get("calories", target_calories)),
                "protein": str(item.get("protein", "90 g")),
                "carbs": str(item.get("carbs", "180 g")),
                "fat": str(item.get("fat", "55 g")),
                "ingredients": item.get("ingredients") if isinstance(item.get("ingredients"), list) else [],
                "instructions": item.get("instructions", ""),
            })

        if len(normalized) < 7:
            template = self._weekly_template(preferences)
            normalized.extend(template[len(normalized):7])
        return normalized[:7]

    def _weekly_template(self, preferences: dict[str, Any]) -> list[dict[str, Any]]:
        diet = preferences.get("diet_type", "any")
        target_calories = int(preferences.get("calories") or 1800)

        templates = {
            "vegetarian": [
                "Oats + Fruit + Seeds | Dal-Roti-Salad | Paneer Bhurji Dinner",
                "Poha + Curd | Rajma Rice + Salad | Tofu Stir Fry + Roti",
                "Besan Chilla + Curd | Khichdi + Veg | Paneer Wrap + Soup",
                "Upma + Nuts | Chole + Phulka + Salad | Veg Pulao + Raita",
                "Moong Chilla + Mint Chutney | Dalia Bowl | Palak Paneer + Roti",
                "Idli-Sambar | Quinoa Veg Bowl | Tofu Curry + Brown Rice",
                "Paratha (light oil) + Curd | Mixed Dal Lunch | Light Veg Soup + Wrap",
            ],
            "non-vegetarian": [
                "Egg Omelette + Toast | Chicken Rice Bowl | Grilled Fish + Veg",
                "Boiled Eggs + Fruit | Chicken Curry + Roti | Egg Bhurji + Salad",
                "Greek Yogurt + Nuts | Fish Curry + Rice | Chicken Stir Fry",
                "Paneer-Egg Wrap | Keema + Phulka | Grilled Chicken + Soup",
                "Egg Sandwich + Fruit | Chicken Salad Bowl | Fish + Saute Veg",
                "Oats + Whey | Chicken Biryani (controlled) + Raita | Egg Curry + Roti",
                "Poha + Eggs | Fish + Brown Rice | Light Chicken Soup + Salad",
            ],
            "any": [
                "Protein Breakfast Bowl | Balanced Lunch Plate | Light Dinner Bowl",
                "Egg/Paneer Breakfast | Rice + Protein Lunch | Soup + Wrap Dinner",
                "Oats + Nuts | Dal/Chicken Lunch | Stir-fry Dinner",
                "Sandwich + Fruit | Khichdi/Rice Bowl | Protein + Veg Dinner",
                "Chilla/Omelette | Curry + Roti | Salad + Soup + Protein",
                "Yogurt + Seeds | Quinoa/ Rice Protein Bowl | Light Dinner Combo",
                "Simple Breakfast | Balanced Lunch | Controlled-carb Dinner",
            ],
        }
        day_titles = templates.get(diet, templates["any"])

        weekly: list[dict[str, Any]] = []
        for idx in range(7):
            weekly.append({
                "meal": f"Day {idx + 1}",
                "title": day_titles[idx % len(day_titles)],
                "image": "",
                "calories": str(target_calories),
                "protein": "90 g",
                "carbs": "180 g",
                "fat": "55 g",
                "ingredients": ["Protein source", "Whole grains", "Vegetables", "Healthy fats", "Hydration"],
                "instructions": "Follow portion control, keep protein in each meal, and hydrate through the day.",
            })
        return weekly

    def _call_huggingface_json(self, prompt: str) -> dict[str, Any]:
        if not self.hf_api_key:
            raise ChatbotServiceError("Missing HUGGINGFACE_API_KEY environment variable.")

        content = self._call_huggingface_completion(
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=520,
        )
        return self._extract_json_object(content)

    def _call_huggingface_text(
        self,
        prompt: str,
        system_prompt: str,
        history: list[dict[str, str]] | None = None,
        max_tokens: int = 320,
    ) -> str:
        if not self.hf_api_key:
            raise ChatbotServiceError("Missing HUGGINGFACE_API_KEY environment variable.")

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for entry in history or []:
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role not in {"user", "assistant"}:
                continue
            if not content:
                continue
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})

        return self._call_huggingface_completion(
            messages=messages,
            temperature=0.4,
            max_tokens=max_tokens,
        )

    def _build_model_candidates(self) -> list[str]:
        ordered: list[str] = []
        for model_name in [self.hf_model, *self.FALLBACK_CHAT_MODELS]:
            model = str(model_name or "").strip()
            if not model or model in ordered:
                continue
            ordered.append(model)
        return ordered or [self.DEFAULT_MODEL]

    def _extract_error_text(self, response: requests.Response) -> str:
        try:
            data = response.json()
            err = data.get("error")
            if isinstance(err, dict):
                message = err.get("message") or err.get("code") or ""
                return str(message).strip()
        except Exception:
            pass
        return (response.text or "").strip()

    def _is_model_not_supported_error(self, response: requests.Response) -> bool:
        if response.status_code != 400:
            return False
        error_text = self._extract_error_text(response).lower()
        signals = [
            "not a chat model",
            "model_not_supported",
            "is not supported by any provider",
        ]
        return any(signal in error_text for signal in signals)

    def _call_huggingface_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        if not self.hf_api_key:
            raise ChatbotServiceError("Missing HUGGINGFACE_API_KEY environment variable.")

        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json",
        }
        last_response: requests.Response | None = None
        selected_model: str | None = None

        for model in self.model_candidates:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            response = self.session.post(self.hf_api_url, headers=headers, json=payload, timeout=self.timeout)
            last_response = response

            if response.ok:
                data = response.json()
                content = str(data["choices"][0]["message"]["content"]).strip()
                selected_model = model
                break

            if self._is_model_not_supported_error(response):
                continue

            response.raise_for_status()
            raise ChatbotServiceError(self._extract_error_text(response) or "Hugging Face request failed.")

        else:
            message = self._extract_error_text(last_response) if last_response is not None else "No model candidate succeeded."
            raise ChatbotServiceError(message or "No model candidate succeeded.")

        if selected_model and selected_model != self.hf_model:
            self.hf_model = selected_model

        if not content:
            raise ChatbotServiceError("Received empty response from language model.")
        return content

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

    def _classify_query(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        if self._is_food_followup(query, history):
            return "food"

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
                if category == "general" and self._is_food_followup(query, history):
                    return "food"
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
            "breakfast ideas", "lunch ideas", "dinner ideas", "snack ideas",
            "kya khau", "kya khaun", "diet chart", "diabetic diet", "sugar patient",
            "pre workout meal", "preworkout meal", "post workout meal", "healthy dinner"
        ]
        if any(phrase in text for phrase in strong_food_phrases):
            return True

        food_keywords = [
            "diet", "recipe", "recipes", "calorie", "calories", "protein", "carbs", "fat",
            "vegetarian", "non veg", "non-veg", "breakfast", "lunch", "dinner", "snack",
            "food", "foods", "meal", "nutrition", "paneer", "chicken", "egg", "dal",
            "diabetes", "diabetic", "sugar", "insulin", "glycemic", "khaun", "khau", "khana", "seb",
            "apple", "banana", "kela", "mango", "aam", "pcos", "thyroid", "preworkout", "postworkout", "gym"
        ]
        food_keyword_hits = sum(1 for keyword in food_keywords if keyword in text)
        number_hits = len(re.findall(r"\b\d+\b", text))

        return food_keyword_hits >= 2 or (food_keyword_hits >= 1 and number_hits >= 1)

    def _is_meal_plan_request(self, query: str, history: list[dict[str, str]] | None = None) -> bool:
        text = query.lower().strip()
        if not text:
            return False

        plan_phrases = [
            "meal plan", "diet plan", "diet chart", "meal chart", "plan banao", "plan bana do",
            "create plan", "generate plan", "full day plan", "weekly plan", "7 day plan",
            "schedule bana", "breakfast lunch dinner plan", "report banao"
        ]
        if any(phrase in text for phrase in plan_phrases):
            return True

        if "plan" in text and any(word in text for word in [
            "meal", "diet", "calorie", "kcal", "weight", "fat loss", "muscle", "gain", "vegetarian", "non veg"
        ]):
            return True

        has_meal_count = bool(re.search(r"\b[1-5]\s*(?:meal|meals)\b", text))
        has_planning_signal = any(word in text for word in [
            "calorie", "kcal", "breakfast", "lunch", "dinner", "snack", "menu", "meal", "plan",
            "weight", "fat loss", "muscle", "gain"
        ])
        if has_meal_count and has_planning_signal:
            return True

        if history and len(text) <= 40:
            if any(word in text for word in ["plan", "chart", "meals", "meal"]) and self._has_recent_food_context(history):
                return True

        return False

    def _is_weekly_plan_request(self, query: str) -> bool:
        text = query.lower().strip()
        weekly_signals = [
            "weekly", "7 day", "seven day", "7-day", "week plan", "full week", "for a week"
        ]
        return any(signal in text for signal in weekly_signals)

    def _generate_chat_reply(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        if not self.hf_api_key:
            return self._local_general_reply(query, history=history)

        context = self._history_context_snippet(history)
        prompt = (
            "Answer the user's question clearly and accurately. "
            "You are a strong general-purpose AI assistant inside a fitness app, but you are not restricted to food topics. "
            "If the user asks non-food topics, answer directly and helpfully. "
            "If the user asks health or nutrition guidance, include practical and safe advice and avoid diagnosis claims. "
            "Use English or Hinglish matching the user style. "
            "Keep formatting simple plain text. "
            "If this is a short follow-up, use conversation context to resolve references.\n"
            f"Recent conversation: {context or 'None'}\n"
            f"User message: {query}"
        )
        try:
            return self._call_huggingface_text(
                prompt,
                (
                    "You are a knowledgeable, friendly assistant. "
                    "Be concise but complete. "
                    "If confidence is low, mention uncertainty briefly. "
                    "Never output markdown code fences. "
                    "Avoid generic filler lines."
                ),
                history=history,
                max_tokens=420,
            )
        except Exception:
            return self._local_general_reply(query, history=history)

    def _generate_dietician_reply(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        if self._prefer_local_dietician_template(query, history):
            return self._local_dietician_reply(query, history=history)

        if not self.hf_api_key:
            return self._local_dietician_reply(query, history=history)

        context = self._history_context_snippet(history)
        prompt = (
            "You are an experienced dietician assistant in a chat app. "
            "The user asked a food/nutrition question and expects practical, personalized advice. "
            "Answer like a real dietician in simple language (English or Hinglish as per user style). "
            "Rules: "
            "1. Give direct answer first. "
            "2. Include practical quantities/portions (e.g., cups, roti count, fruit count, grams). "
            "3. If user has diabetes/sugar concern, suggest low glycemic choices and portion control. "
            "4. If user asks about a single food (e.g., apple), give daily safe quantity and best timing. "
            "5. Keep response concise but actionable, plain text only. "
            "6. Include a short safety note for medication/insulin users when relevant. "
            "7. If this is a follow-up, continue from previous context without repeating full lecture. "
            f"Recent conversation: {context or 'None'}. "
            f"User message: {query}"
        )
        try:
            return self._call_huggingface_text(
                prompt,
                (
                    "You are a practical clinical dietician coach. "
                    "Give specific food advice with quantities and timing. "
                    "No markdown code fences. "
                    "Avoid vague, generic responses."
                ),
                history=history,
                max_tokens=520,
            )
        except Exception:
            return self._local_dietician_reply(query, history=history)

    def _prefer_local_dietician_template(self, query: str, history: list[dict[str, str]] | None = None) -> bool:
        text = query.lower().strip()
        if not text:
            return True

        direct_condition_terms = [
            "diabetes", "diabetic", "sugar", "apple", "seb", "banana", "kela", "mango", "aam",
            "pcos", "thyroid", "preworkout", "pre workout", "before gym", "dinner", "night meal"
        ]
        if any(term in text for term in direct_condition_terms):
            return True

        if history and len(text) <= 40 and self._has_recent_food_context(history):
            return True

        return False

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

    def _safe_float(self, value: Any) -> float:
        if value is None or value == "":
            return 0.0
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        if not match:
            return 0.0
        try:
            return float(match.group(0))
        except (TypeError, ValueError):
            return 0.0

    def _sanitize_history(self, history: Any) -> list[dict[str, str]]:
        if not isinstance(history, list):
            return []

        cleaned: list[dict[str, str]] = []
        for item in history[-10:]:
            if not isinstance(item, dict):
                continue

            role_raw = str(item.get("role", "")).strip().lower()
            if role_raw in {"assistant", "bot"}:
                role = "assistant"
            elif role_raw == "user":
                role = "user"
            else:
                continue

            content = str(item.get("content") or item.get("text") or "").strip()
            if not content:
                continue

            cleaned.append({"role": role, "content": content[:1200]})

        return cleaned

    def _has_recent_food_context(self, history: list[dict[str, str]] | None) -> bool:
        if not history:
            return False

        joined = " ".join(item.get("content", "").lower() for item in history[-8:])
        if not joined.strip():
            return False

        food_signals = [
            "diet", "meal", "calorie", "protein", "carb", "fat", "food", "eat", "breakfast", "lunch", "dinner",
            "snack", "vegetarian", "non veg", "recipe", "diabetic", "diabetes", "sugar", "pcos", "thyroid",
            "weight loss", "weight gain", "fat loss", "muscle gain", "apple", "banana", "fruit"
        ]
        return any(signal in joined for signal in food_signals)

    def _is_food_followup(self, query: str, history: list[dict[str, str]] | None) -> bool:
        text = query.lower().strip()
        if not text:
            return False

        if self._fallback_food_detection(text):
            return True

        if not self._has_recent_food_context(history):
            return False

        followup_cues = [
            "aur", "also", "what about", "kya", "kitna", "kitne", "kitni", "can i", "should i", "timing",
            "portion", "quantity", "breakfast", "lunch", "dinner", "snack", "fruit", "apple", "banana", "mango",
            "preworkout", "post workout", "veg", "vegetarian", "non veg", "option", "meal", "diet"
        ]
        if len(text) <= 60 and any(cue in text for cue in followup_cues):
            return True

        return False

    def _history_context_snippet(self, history: list[dict[str, str]] | None, limit: int = 4) -> str:
        if not history:
            return ""

        parts: list[str] = []
        for item in history[-limit:]:
            role = "User" if item.get("role") == "user" else "Assistant"
            content = (item.get("content") or "").strip()
            if not content:
                continue
            parts.append(f"{role}: {content}")
        return " | ".join(parts)

    def _extract_goal_focus(self, query: str) -> str:
        text = query.lower()
        if any(word in text for word in ["weight loss", "lose", "fat loss", "cut"]):
            return "Weight Loss"
        if any(word in text for word in ["gain", "bulk", "muscle", "build"]):
            return "Muscle Gain"
        if any(word in text for word in ["diabetes", "sugar", "pcos", "thyroid", "bp"]):
            return "Condition Support"
        return "Balanced Nutrition"

    def _build_report_snapshot(
        self,
        query: str,
        meal_plan: list[dict[str, Any]],
        preferences: dict[str, Any],
    ) -> dict[str, Any]:
        total_calories = round(sum(self._safe_float(item.get("calories")) for item in meal_plan))
        total_protein = round(sum(self._safe_float(item.get("protein")) for item in meal_plan), 1)
        total_carbs = round(sum(self._safe_float(item.get("carbs")) for item in meal_plan), 1)
        total_fat = round(sum(self._safe_float(item.get("fat")) for item in meal_plan), 1)

        calories_from_macros = max((total_protein * 4) + (total_carbs * 4) + (total_fat * 9), 1)
        protein_pct = round((total_protein * 4 / calories_from_macros) * 100)
        carbs_pct = round((total_carbs * 4 / calories_from_macros) * 100)
        fat_pct = round((total_fat * 9 / calories_from_macros) * 100)

        target_calories = int(preferences.get("calories") or 0)
        calorie_delta = total_calories - target_calories if target_calories else 0
        hydration_target = "2.5-3.5 L/day" if target_calories >= 2200 else "2.0-3.0 L/day"
        goal_focus = self._extract_goal_focus(query)
        diet_label = str(preferences.get("diet_type") or "any").replace("-", " ").title()
        meals_count = len(meal_plan)

        recommendations = [
            f"Keep meal timings consistent across your {meals_count}-meal structure.",
            "Aim for at least 25-35 g protein per major meal for better satiety and recovery.",
            "Use minimally processed foods and include colorful vegetables twice daily.",
            f"Hydration target: {hydration_target}.",
        ]
        if calorie_delta and abs(calorie_delta) > 220:
            if calorie_delta > 0:
                recommendations.append("Portion sizes appear calorie-dense. Slightly reduce oils and refined carbs.")
            else:
                recommendations.append("Calories are below target. Add one dense snack with protein and healthy fats.")

        return {
            "generated_on": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "goal_focus": goal_focus,
            "diet_type": diet_label,
            "requested_meals": int(preferences.get("meals") or meals_count or 0),
            "daily_target_calories": target_calories,
            "total_calories": total_calories,
            "calorie_gap": calorie_delta,
            "total_protein_g": total_protein,
            "total_carbs_g": total_carbs,
            "total_fat_g": total_fat,
            "macro_split": {
                "protein_pct": protein_pct,
                "carbs_pct": carbs_pct,
                "fat_pct": fat_pct,
            },
            "hydration_target": hydration_target,
            "recommendations": recommendations[:6],
            "notes": (
                "This plan is educational and should be personalized for medical conditions, allergies, "
                "medications, and lab values with a licensed dietitian."
            ),
        }

    def _local_general_reply(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        text = query.lower().strip()
        if not text:
            return "Please type your question and I will help."

        history_text = self._history_context_snippet(history, limit=6).lower()

        if history_text:
            if any(key in history_text for key in ["list comprehension", "python"]) and any(
                key in text for key in ["example", "simple", "hindi", "samjhao", "explain"]
            ):
                if "hindi" in text or "samjhao" in text:
                    return (
                        "List comprehension ka matlab: ek hi line me list banana. "
                        "Example: numbers = [1,2,3,4,5], squares = [x*x for x in numbers] "
                        "to result [1,4,9,16,25] aata hai."
                    )
                return (
                    "Simple example: evens = [x for x in range(1,11) if x % 2 == 0]. "
                    "Result: [2,4,6,8,10]."
                )

            if any(key in history_text for key in ["sleep", "stress", "anxious", "anxiety"]) and any(
                key in text for key in ["more", "aur", "routine", "daily", "plan", "example"]
            ):
                return (
                    "Daily anxiety/sleep mini-routine: "
                    "Morning 10 min sunlight + walk, afternoon caffeine limit, evening light dinner, "
                    "night 10 min breathing + no screens 45 min before sleep."
                )

        if text in {"hi", "hello", "hey", "hii", "namaste"}:
            return (
                "Hi! Main general questions, personal guidance, aur dietician-style food advice sab de sakta hoon. "
                "Aap jo poochna chaho seedha poochho."
            )

        if any(phrase in text for phrase in ["who are you", "what can you do"]):
            return (
                "I can answer general questions, explain topics clearly, and also generate structured diet meal plans. "
                "For meal plans, mention calories, goal, diet type, and meals per day."
            )

        if re.fullmatch(r"[0-9+\-*/().\s]{3,40}", text):
            try:
                result = eval(text, {"__builtins__": {}}, {})
                return f"{query.strip()} = {result}"
            except Exception:
                pass

        if "2+2" in text or text == "what is 2 + 2" or text == "what is 2+2":
            return "2 + 2 = 4."

        if "list comprehension" in text and "python" in text:
            return (
                "Python list comprehension ek short way hai list banane ka. "
                "Example: squares = [x*x for x in range(1,6)] output [1,4,9,16,25]. "
                "Yani loop + condition ko one line me likh sakte ho."
            )

        if any(word in text for word in ["sleep", "stress", "anxious", "anxiety"]):
            return (
                "Simple routine try karo: fixed sleep time, sone se 1 ghanta pehle screen band, "
                "shaam ke baad caffeine kam, 10-minute breathing, aur daily 20-30 min walk. "
                "Agar anxiety regularly disturb kare to mental health professional se baat karna best rahega."
            )

        if any(word in text for word in ["diet", "meal", "nutrition", "protein", "calorie"]):
            return (
                "Main dietician-style advice bhi de sakta hoon aur meal plan bhi bana sakta hoon. "
                "Example: \"Main diabetic hoon, kya khaaun?\" ya \"4 meal 1800 calorie fat loss plan banao\"."
            )

        return "I can help with this. Please share a little more detail so I can give a precise answer."

    def _local_dietician_reply(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        text = query.lower().strip()
        if not text:
            return "Aap apna diet question type karo, main practical portion ke saath guide karunga."

        if history and len(text) <= 60 and self._has_recent_food_context(history):
            previous_user = " ".join(
                item.get("content", "").lower()
                for item in history[-8:]
                if item.get("role") == "user"
            )
            if any(term in previous_user for term in ["diabetes", "diabetic", "sugar", "blood sugar", "glucose", "madhumeh"]):
                return self._local_diabetes_reply(f"{text} diabetic")
            if any(term in previous_user for term in ["weight loss", "fat loss", "lose"]) and any(
                term in text for term in ["veg", "vegetarian", "option", "dinner", "lunch", "breakfast"]
            ):
                return (
                    "Weight loss ke liye veg option: "
                    "Breakfast: oats + dahi + seeds, "
                    "Lunch: 2 phulka + dal + salad + sabzi, "
                    "Dinner: paneer/tofu bhurji + stir-fry sabzi + 1 phulka. "
                    "Har meal me protein include karo."
                )

        diabetes_terms = ["diabetes", "diabetic", "sugar", "blood sugar", "glucose", "madhumeh"]
        if any(term in text for term in diabetes_terms):
            return self._local_diabetes_reply(text)

        if any(term in text for term in ["apple", "seb"]):
            return (
                "Apple generally healthy hai. Ek serving = 1 small-medium apple (100-120 g). "
                "Agar weight loss ya sugar concern hai to 1 apple ek time par lo, juice avoid karo, "
                "aur better hai ki nuts/curd ke saath lo for stable energy."
            )

        if any(term in text for term in ["banana", "kela"]):
            return (
                "Banana le sakte ho, but portion control zaroori hai. "
                "Diabetes/weight loss me 1 small banana (ya half large) ek time par lo, preferably workout se pehle "
                "ya morning snack me, aur protein source (nuts/curd) ke saath lo."
            )

        if any(term in text for term in ["mango", "aam"]):
            return (
                "Diabetes me mango completely ban nahi hai, par controlled portion lo: 1/2 cup mango cubes "
                "(about 80-100 g) ek time par, daily nahi. Juice/shake avoid karo aur protein/fiber ke saath lo."
            )

        if any(term in text for term in ["preworkout", "pre workout", "before gym"]):
            return (
                "Pre-workout (45-90 min pehle): 1 banana + 5-6 almonds, ya 1 toast + peanut butter, "
                "ya dahi + fruit. Heavy oily food avoid karo. "
                "Agar intense training hai to small carb + protein combo best rahega."
            )

        if any(term in text for term in ["dinner", "raat", "night meal"]):
            return (
                "Dinner simple rakho: 1 bowl sabzi + protein (dal/paneer/chicken/fish) + "
                "1-2 phulka ya small rice portion. Sone se 2-3 ghante pehle dinner complete karo."
            )

        if any(term in text for term in ["pcos"]):
            return (
                "PCOS ke liye har meal me protein + fiber rakho: eggs/paneer/dal + sabzi + controlled carbs. "
                "Sugary drinks aur refined snacks kam karo, aur 30-40 min daily activity rakho. "
                "Agar chaho to main one-day PCOS meal structure de deta hoon."
            )

        if any(term in text for term in ["thyroid"]):
            return (
                "Thyroid ke liye balanced diet rakho: adequate protein, vegetables, whole grains, nuts/seeds. "
                "Medication agar lete ho to tablet empty stomach lo aur calcium/iron supplements se gap rakho "
                "(doctor guidance ke mutabik)."
            )

        return (
            "Dietician tip: Har meal me protein + fiber rakho. "
            "Simple plate rule: 1/2 plate sabzi, 1/4 protein (dal/paneer/egg/chicken), "
            "1/4 complex carbs (1-2 roti ya 1 cup chawal). "
            "Agar chaho to main isi goal ke hisab se exact one-day plan bana deta hoon."
        )

    def _local_diabetes_reply(self, text: str) -> str:
        if any(term in text for term in ["apple", "seb"]):
            return (
                "Agar aap diabetic ho, to apple le sakte ho: 1 small-medium apple (100-120 g) ek baar me. "
                "Din me 1-2 apples max rakho, lekin ek saath 2 mat lo. "
                "Best timing: mid-morning ya evening, aur nuts/peanut/curd ke saath lo. "
                "Apple juice avoid karo. Meal ke 2 ghante baad sugar check karke apni tolerance dekho."
            )

        if any(term in text for term in ["banana", "kela"]):
            return (
                "Diabetes me banana possible hai, but controlled: 1 small banana ya 1/2 large banana ek time par. "
                "Daily zaroori nahi. Isko nuts/dahi ke saath lo aur sugar response monitor karo."
            )

        if any(term in text for term in ["mango", "aam"]):
            return (
                "Diabetes me mango occasional portion me le sakte ho: 1/2 cup (80-100 g) mango cubes. "
                "Juice/pulp based drinks avoid karo. Mango ko empty stomach na lo; protein/fiber ke saath lo."
            )

        if any(term in text for term in ["breakfast", "morning", "subah"]):
            return (
                "Diabetes-friendly breakfast options: "
                "1) Vegetable besan chilla (2) + curd, "
                "2) Veg omelette + 1 multigrain toast, "
                "3) Oats + chia + nuts (no sugar), "
                "4) Moong dal chilla + salad. "
                "Target: breakfast me protein + fiber zaroor rakho."
            )

        if any(term in text for term in ["dinner", "raat", "night"]):
            return (
                "Diabetes dinner option: 1 bowl non-starchy sabzi + protein (dal/paneer/fish/chicken) + "
                "1-2 phulka ya 1/2-1 cup brown rice. Dinner light rakho aur sone se 2-3 ghante pehle finish karo."
            )

        return (
            "Diabetes ke liye practical diet:\n"
            "1. Har meal me plate method use karo: 1/2 non-starchy sabzi, 1/4 protein, 1/4 complex carbs.\n"
            "2. Carbs portion: 1-2 phulka ya 1 cup cooked rice per meal (individual sugar response ke hisab se adjust).\n"
            "3. Fruits: ek time par 1 serving (apple/guava/orange), juice nahi.\n"
            "4. Snacks: roasted chana, sprouts, paneer, curd, nuts.\n"
            "5. Avoid: sugary drinks, sweets, bakery snacks, refined maida.\n"
            "Agar aap insulin/medicine par ho, to meal timing regular rakho aur doctor ke plan ke saath align karo."
        )

    def _clean_text(self, value: str) -> str:
        if not value:
            return ""
        text = re.sub(r"<[^>]+>", " ", unescape(value))
        text = re.sub(r"\s+", " ", text).strip()
        return text
