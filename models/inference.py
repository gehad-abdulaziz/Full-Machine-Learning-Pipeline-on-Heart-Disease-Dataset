import pandas as pd
import joblib

pipeline = joblib.load('./models/best_model_pipeline.pkl')

df_new = pd.read_csv('./Data/cleaned_heart_disease.csv')

print("Columns in new data:", df_new.columns.tolist())

col_name = df_new.columns[9] 
X_new = df_new.drop(col_name, axis=1)

predictions = pipeline.predict(X_new)

print("✅ Predictions:\n", predictions)

output_df = pd.DataFrame(predictions, columns=['Prediction'])
output_df.to_csv('./Data/predictions.csv', index=False)
print("🎯✅ Predictions saved to ./Data/predictions.csv")
