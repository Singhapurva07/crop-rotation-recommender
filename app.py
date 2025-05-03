from flask import Flask, request, render_template
import numpy as np
import joblib
from dotenv import load_dotenv
import os
import google.generativeai as genai

app = Flask(__name__)

# Enable Flask request logging
app.logger.setLevel('DEBUG')

try:
    print("Loading environment variables...")
    # Load environment variables from .env file
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    print("GEMINI_API_KEY loaded successfully")

    print("Initializing Gemini 2.0 Flash model...")
    # Initialize the Gemini 2.0 Flash model
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    print("Gemini model initialized successfully")

    print("Loading trained crop prediction model and label encoder...")
    # Load the trained crop prediction model and label encoder
    model = joblib.load('crop_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("Model and label encoder loaded successfully")

    # Define crop families and nitrogen-fixing crops (same as in train.py)
    CROP_FAMILIES = {
        'rice': 'Poaceae', 'maize': 'Poaceae', 'wheat': 'Poaceae', 'millet': 'Poaceae',
        'chickpea': 'Fabaceae', 'lentil': 'Fabaceae', 'pigeonpeas': 'Fabaceae', 'mothbeans': 'Fabaceae',
        'mungbean': 'Fabaceae', 'blackgram': 'Fabaceae', 'kidneybeans': 'Fabaceae', 'peas': 'Fabaceae',
        'cotton': 'Malvaceae', 'coffee': 'Rubiaceae', 'jute': 'Malvaceae', 'coconut': 'Arecaceae',
        'banana': 'Musaceae', 'mango': 'Anacardiaceae', 'apple': 'Rosaceae', 'papaya': 'Caricaceae',
        'orange': 'Rutaceae', 'pomegranate': 'Lythraceae', 'grapes': 'Vitaceae', 'watermelon': 'Cucurbitaceae',
        'muskmelon': 'Cucurbitaceae'
    }

    NITROGEN_FIXING_CROPS = [
        'chickpea', 'lentil', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'kidneybeans', 'peas'
    ]

    PROFITABILITY_SCORES = {
        'rice': 0.5, 'maize': 0.4, 'wheat': 0.45, 'millet': 0.35,
        'chickpea': 0.6, 'lentil': 0.65, 'pigeonpeas': 0.55, 'mothbeans': 0.5,
        'mungbean': 0.55, 'blackgram': 0.5, 'kidneybeans': 0.6, 'peas': 0.65,
        'cotton': 0.8, 'coffee': 1.2, 'jute': 0.7, 'coconut': 0.9,
        'banana': 0.7, 'mango': 1.0, 'apple': 1.1, 'papaya': 0.75,
        'orange': 0.8, 'pomegranate': 0.95, 'grapes': 1.0, 'watermelon': 0.6,
        'muskmelon': 0.65
    }

    # List of crops for the dropdown
    CROPS = sorted(CROP_FAMILIES.keys())

    @app.route('/')
    def home():
        print("Rendering home page...")
        # If the form was submitted, retrieve the previous crop; otherwise, set to None
        previous_crop = request.args.get('previous_crop', None)
        return render_template('index.html', crops=CROPS, previous_crop=previous_crop)

    @app.route('/recommend', methods=['POST'])
    def recommend():
        app.logger.debug("Received POST request to /recommend")
        print("Processing recommendation request...")
        # Get input data from the form
        try:
            previous_crop = request.form['previous_crop']
            if previous_crop not in CROP_FAMILIES:
                raise ValueError("Invalid previous crop selected")
            nitrogen = float(request.form['nitrogen'])
            phosphorus = float(request.form['phosphorus'])
            potassium = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            print(f"Previous crop: {previous_crop}")
            print(f"Input data: N={nitrogen}, P={phosphorus}, K={potassium}, Temp={temperature}, "
                  f"Humidity={humidity}, pH={ph}, Rainfall={rainfall}")
        except KeyError as e:
            error_msg = f"Form input error: Missing field {e}"
            print(error_msg)
            app.logger.error(error_msg)
            return render_template('index.html', crops=CROPS, previous_crop=None, error=error_msg), 400
        except ValueError as e:
            error_msg = f"Form input error: Invalid value {e}"
            print(error_msg)
            app.logger.error(error_msg)
            return render_template('index.html', crops=CROPS, previous_crop=previous_crop, error=error_msg), 400

        # Prepare input for the crop prediction model
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

        # Predict the crop
        print("Predicting crop...")
        try:
            prediction = model.predict(input_data)
            predicted_crop = label_encoder.inverse_transform(prediction)[0]
            print(f"Initial predicted crop: {predicted_crop}")
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(error_msg)
            app.logger.error(error_msg)
            return render_template('index.html', crops=CROPS, previous_crop=previous_crop, error=error_msg), 500

        # Apply crop rotation logic
        previous_family = CROP_FAMILIES.get(previous_crop, '')
        predicted_family = CROP_FAMILIES.get(predicted_crop, '')
        is_nitrogen_fixing = predicted_crop in NITROGEN_FIXING_CROPS

        # Explain why this crop is a good alternation
        alternation_reason = []
        if previous_family and predicted_family and previous_family != predicted_family:
            alternation_reason.append(f"{predicted_crop} belongs to the {predicted_family} family, which is different from the previous crop's family ({previous_family}). This reduces the risk of pest and disease buildup.")
        if previous_crop not in NITROGEN_FIXING_CROPS and is_nitrogen_fixing:
            alternation_reason.append(f"{predicted_crop} is a nitrogen-fixing crop, which will help replenish soil nitrogen depleted by the previous crop ({previous_crop}).")
        elif previous_crop in NITROGEN_FIXING_CROPS and not is_nitrogen_fixing:
            alternation_reason.append(f"The previous crop ({previous_crop}) was nitrogen-fixing, so {predicted_crop} can take advantage of the enriched soil nitrogen.")
        alternation_explanation = " ".join(alternation_reason) if alternation_reason else "This crop is suitable based on soil and weather conditions."

        # Check if the predicted crop is from the same family as the previous crop
        if previous_family and predicted_family and previous_family == predicted_family:
            print("Same family detected, applying rotation logic...")
            # If same family, try to recommend a nitrogen-fixing crop or a different family
            alternative_crops = [crop for crop in label_encoder.classes_ if CROP_FAMILIES.get(crop, '') != previous_family]
            if alternative_crops:
                # Prefer nitrogen-fixing crops if the previous crop was not nitrogen-fixing
                if previous_crop not in NITROGEN_FIXING_CROPS:
                    nitrogen_fixing_alternatives = [crop for crop in alternative_crops if crop in NITROGEN_FIXING_CROPS]
                    if nitrogen_fixing_alternatives:
                        predicted_crop = max(nitrogen_fixing_alternatives, key=lambda x: PROFITABILITY_SCORES.get(x, 0))
                        print(f"Switched to nitrogen-fixing crop: {predicted_crop}")
                        alternation_explanation = (f"{predicted_crop} was chosen because it belongs to a different family ({CROP_FAMILIES[predicted_crop]}) "
                                                  f"and is nitrogen-fixing, helping to replenish soil nitrogen after {previous_crop}.")
                    else:
                        predicted_crop = max(alternative_crops, key=lambda x: PROFITABILITY_SCORES.get(x, 0))
                        print(f"Switched to different family crop: {predicted_crop}")
                        alternation_explanation = (f"{predicted_crop} was chosen because it belongs to a different family ({CROP_FAMILIES[predicted_crop]}) "
                                                  f"than {previous_crop}, reducing pest and disease risks.")
                else:
                    predicted_crop = max(alternative_crops, key=lambda x: PROFITABILITY_SCORES.get(x, 0))
                    print(f"Switched to different family crop: {predicted_crop}")
                    alternation_explanation = (f"{predicted_crop} was chosen because it belongs to a different family ({CROP_FAMILIES[predicted_crop]}) "
                                              f"than {previous_crop}, reducing pest and disease risks.")

        # Get profitability score
        profitability_score = PROFITABILITY_SCORES.get(predicted_crop, 0)
        print(f"Final recommended crop: {predicted_crop}, Profitability Score: {profitability_score}")

        # Explain benefits to the farmer
        farmer_benefits = (f"Planting {predicted_crop} can be profitable with a score of {profitability_score}. "
                           f"It is well-suited to the current soil and weather conditions, potentially leading to high yields.")

        # Explain benefits to the soil
        soil_benefits = []
        if is_nitrogen_fixing:
            soil_benefits.append(f"{predicted_crop} fixes nitrogen in the soil, improving fertility for future crops.")
        if previous_family != predicted_family:
            soil_benefits.append("Rotating crop families helps break pest and disease cycles, maintaining soil health.")
        soil_benefits.append("This rotation ensures balanced nutrient use, preventing soil depletion.")
        soil_benefits_explanation = " ".join(soil_benefits)

        # Use Gemini 2.0 Flash to generate detailed suggestions
        print("Generating detailed suggestions with Gemini 2.0 Flash...")
        prompt = (f"Provide detailed suggestions for growing {predicted_crop} in a field with the following conditions: "
                  f"Nitrogen: {nitrogen} kg/ha, Phosphorus: {phosphorus} kg/ha, Potassium: {potassium} kg/ha, "
                  f"Temperature: {temperature}°C, Humidity: {humidity}%, pH: {ph}, Rainfall: {rainfall} mm. "
                  f"Include planting tips, soil management strategies, and market insights for profitability.")
        try:
            gemini_response = gemini_model.generate_content(prompt)
            detailed_suggestions = gemini_response.text
            print("Suggestions generated successfully")
        except Exception as e:
            detailed_suggestions = f"Error generating suggestions: {str(e)}"
            print(f"Gemini API error: {str(e)}")

        print("Rendering recommendation page...")
        return render_template('index.html', crops=CROPS, previous_crop=previous_crop, recommended_crop=predicted_crop,
                               profitability_score=profitability_score, detailed_suggestions=detailed_suggestions,
                               alternation_explanation=alternation_explanation, farmer_benefits=farmer_benefits,
                               soil_benefits=soil_benefits_explanation)

except Exception as e:
    print(f"Failed to start application: {str(e)}")
    raise

if __name__ == '__main__':
    try:
        print("Starting Flask server on port 5002...")
        app.run(debug=True, port=5002, use_reloader=False)
    except Exception as e:
        print(f"Flask server failed to start: {str(e)}")