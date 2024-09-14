import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Title of the app
st.title("Product Review Rating Predictor")

# Upload dataset option
st.header("Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded Successfully:")
    st.write(data.head())

    # Step 2: Preprocessing the data
    st.header("Step 2: Preprocessing the Data")
    
    # Check for required columns 'review' and 'rating'
    if 'review' in data.columns and 'rating' in data.columns:
        X = data['review']  # Features (review text)
        y = data['rating']  # Labels (rating)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(stop_words='english', max_df=0.95)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        # Train the Naive Bayes model
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)

        # Step 3: User Input
        st.header("Step 3: Enter a Review for Prediction")
        user_input = st.text_area("Enter a product review:", "This product is amazing!")
        
        if st.button("Predict Rating"):
            if user_input:
                # Preprocess user input
                input_tfidf = tfidf.transform([user_input])
                
                # Predict the rating
                prediction = model.predict(input_tfidf)[0]
                
                # Display the predicted rating
                st.write(f"Predicted Rating: {prediction}/5")
            else:
                st.write("Please enter a review for prediction.")

        # Show model performance (Optional)
        if st.checkbox("Show Model Performance"):
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy:.2f}")
            st.write("Classification Report:\n", classification_report(y_test, y_pred))
    else:
        st.write("Error: Please upload a CSV file with 'review' and 'rating' columns.")
else:
    st.write("Please upload a CSV file to proceed.")
