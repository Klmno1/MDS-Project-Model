# Start Server uvicorn main:app --reload --port 8000
# 	This tells Uvicorn where your FastAPI app is located.
# 🔹 main = the filename (main.py)
# 🔹 app = the FastAPI instance inside that file (e.g., app = FastAPI())
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pickle
import pandas as pd
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"],
    allow_origins=["http://localhost:3000"], # Only allow local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input format from Frontend (remove price!)
class InputData(BaseModel):
    customerID: str
    category: str
    product: str
    date: str

# Load models and assets
# with open("Stacking_model.pkl", "rb") as f:
#     stack_model = pickle.load(f)

with open("XGBoost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("category_rules.pkl", "rb") as f:
    category_rules = pickle.load(f)
    
# Load customer-country mapping
customer_country_df = pd.read_csv("customer_country_mapping.csv")
customer_country_df['Customer ID'] = customer_country_df['Customer ID'].astype(str)
customer_to_country = dict(zip(customer_country_df['Customer ID'], customer_country_df['Country']))
country_list = customer_country_df['Country'].dropna().unique().tolist()

# Load product prices
product_price_df = pd.read_csv("product_price_map.csv")
product_price_df['Description'] = product_price_df['Description'].str.strip().str.lower()
product_to_price = dict(zip(product_price_df['Description'], product_price_df['Price']))

# Determine season from month
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [12, 1, 2]:
        return 'Winter'
    else:
        return 'Autumn'

# Prepare input row
def prepare_input_row(data: InputData, rules) -> pd.DataFrame:
    try:
        month = datetime.strptime(data.date, "%Y-%m-%d").month
    except ValueError:
        raise ValueError(f"Invalid date format: {data.date}. Expected YYYY-MM-DD.")
    
    season = get_season(month)
    basket = {data.category}

    default_country = customer_to_country.get(data.customerID, 'Unspecified')
    if default_country is None:
        raise ValueError(f"Country not found for customer ID: {data.customerID}")

    price = product_to_price.get(data.product.lower())
    if price is None:
        raise ValueError(f"Price not found for product: {data.product}")

    row = {
        'Price': price,
        'Customer ID': data.customerID,
        'Month': month,
        'category': data.category
    }

    for s in ['Spring', 'Summer', 'Winter']:
        row[f'Season_{s}'] = int(season == s)

    for c in country_list:
        row[f'Country_{c}'] = int(c == default_country)

    for idx, rule in enumerate(rules):
        lhs_items = set(rule.lhs)
        row[f'cat_rule_{idx:03d}'] = int(lhs_items.issubset(basket))
    
    # for idx, rule in enumerate(rules):
    #     lhs = '_'.join(rule.lhs).replace(" ", "")
    #     rhs = '_'.join(rule.rhs).replace(" ", "")
    #     rule_label = f"{lhs}_to_{rhs}"
    #     row[rule_label] = int(set(rule.lhs).issubset(basket))

    return pd.DataFrame([row]), price

# def clean_feature_name(name):
#     # Remove prefix before double underscores "__"
#     # e.g. "num__Price" -> "Price"
#     #       "cat__category_Interior Finishes" -> "category_Interior Finishes"
#     #       "remainder__Country_Germany" -> "Country_Germany"
#     if "__" in name:
#         return name.split("__", 1)[1]
#     return name

# def get_feature_importance(stack_model, feature_names):
#     # Extract base learners and meta-model
#     xgb = stack_model.named_estimators_['xgb'].named_steps['xgbclassifier']
#     rf = stack_model.named_estimators_['rf'].named_steps['randomforestclassifier']
#     meta_model = stack_model.final_estimator_

#     # Get feature importances from base learners
#     xgb_importance = xgb.feature_importances_
#     rf_importance = rf.feature_importances_

#     # Normalize base importances
#     xgb_norm = xgb_importance / xgb_importance.sum()
#     rf_norm = rf_importance / rf_importance.sum()

#     # # Weight base importances by logistic regression coefficients
#     # # These are the meta-model weights for xgb and rf predictions
#     # meta_weights = meta_model.coef_[0]
#     # xgb_weight = meta_weights[0]
#     # rf_weight = meta_weights[1]
    
#     # # Zip with feature names
#     # importance_dict = dict(zip(feature_names, final_importance))

#     # # Combined weighted feature importance
#     # final_importance = xgb_weight * xgb_norm + rf_weight * rf_norm
    
#     # Average the normalized importances
#     avg_importance = (xgb_norm + rf_norm) / 2
    
#     # Clean feature names
#     cleaned_feature_names = [clean_feature_name(f) for f in feature_names]
    
#     importance_dict = dict(zip(cleaned_feature_names, avg_importance))

#     # Sort by importance
#     sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

#     return sorted_importance

# def get_feature_importance(xgb_model, feature_names):
#     # For XGBoost model, feature_importances_ attribute exists
#     importance_values = xgb_model.feature_importances_

#     cleaned_feature_names = [clean_feature_name(f) for f in feature_names]
#     importance_dict = dict(zip(cleaned_feature_names, importance_values))

#     sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
#     return sorted_importance

# XGB Only
def get_feature_importance(xgb_model, feature_names):
    # For XGBoost model, feature_importances_ attribute exists
    importance_values = xgb_model.named_steps['xgbclassifier'].feature_importances_

    importance_dict = dict(zip(feature_names, importance_values))

    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    return sorted_importance

@app.post("/predict")
async def predict(data: InputData):
    try:
        input_df, price = prepare_input_row(data, category_rules)
        print(input_df)
        # Model input:
        # Price, CustomerID, Month, Country, Season, category, cat_rule for input
        # prob return probs of class [class 0, class 1] where class 1 is the returned prob
        # prob = stack_model.predict_proba(input_df)[0][1]
        prob = xgb_model.predict_proba(input_df)[0][0]
        
        # feature_names = stack_model.named_estimators_['xgb'].named_steps['preprocessor'].get_feature_names_out()
        # importances = get_feature_importance(stack_model, feature_names)
        
        importances = get_feature_importance(xgb_model, list(input_df.columns))
        
        # Step 1: Build mapping from cat_rule_XXX to readable rule name
        rule_name_map = {}
        for idx, rule in enumerate(category_rules):
            lhs = '_'.join(rule.lhs).replace(" ", "")
            rhs = '_'.join(rule.rhs).replace(" ", "")
            rule_label = f"{lhs}_to_{rhs}"
            rule_name_map[f'cat_rule_{idx:03d}'] = rule_label

        # Step 2: Replace keys in importances
        readable_importances = {}
        for key, val in importances.items():
            readable_key = rule_name_map.get(key, key)  # fallback to original if not a cat_rule
            readable_importances[readable_key] = val

        # Optional: sort by importance descending
        readable_importances = dict(sorted(readable_importances.items(), key=lambda x: -x[1]))
        
        # print(readable_importances)
        return {
            "price": round(float(price), 2),
            "category": data.category,
            "product_name": data.product,
            "date": data.date,
            "probability": round(float(prob), 4),
            "feature_importance": {k: float(v) for k, v in readable_importances.items()}
            # "feature_importance": dict(list(importances.items())[:10]),  # return top 10
        }
    except Exception as e:
        return {"error": str(e)}
