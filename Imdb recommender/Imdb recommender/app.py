# app.py
import streamlit as st
from recommender import load_data, get_recommendations

st.title("🎬 IMDb Movie Recommender (2024)")
st.write("Enter a short storyline and get similar movie suggestions!")

storyline_input = st.text_area("Enter storyline here:")

if st.button("Get Recommendations"):
    if storyline_input.strip() == "":
        st.warning("Please enter a storyline to proceed.")
    else:
        df = load_data()
        recommendations = get_recommendations(storyline_input, df)

        st.subheader("Top 5 Recommended Movies:")
        for i, row in recommendations.iterrows():
            st.markdown(f"**{row['Movie Name']}**")
            st.write(row['Storyline'])
