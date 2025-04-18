import pandas as pd
import streamlit as st

# Function to load data
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

# Streamlit UI
st.title("Patron Management Tool")

# Default file path
default_file = "Patrons.csv"  # Replace with your default file path

# File uploader
#uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Load the default file if no file is uploaded
#if uploaded_file:
#    data = load_data(uploaded_file)
#else:
#    st.write(f"No file uploaded. Using default file: {default_file}")
#    data = load_data(default_file)
data = load_data(default_file)

# Display the data
#st.write("### Raw Data", data)

# Filter options
st.sidebar.title("Filter Options")
for col in data.columns:
    unique_values = data[col].dropna().unique()
    if len(unique_values) <= 20:  # Dropdown for categorical columns
        selected = st.sidebar.multiselect(f"Filter {col}", unique_values)
        if selected:
            data = data[data[col].isin(selected)]

# Sorting options
st.sidebar.title("Sorting Options")
sort_column = st.sidebar.selectbox("Sort by", data.columns)
ascending = st.sidebar.radio("Order", ["Ascending", "Descending"]) == "Ascending"
if sort_column:
    data = data.sort_values(by=sort_column, ascending=ascending)

# Display filtered/sorted data
st.write("### Filtered/Sorted Data", data)

# Download button
st.download_button("Download Filtered Data", data.to_csv(index=False), "filtered_data.csv", "text/csv")