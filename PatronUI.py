import pandas as pd
import streamlit as st

# Cache the loading function for efficiency
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        elif file.name.endswith('.csv'):
            return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

# Title and description
st.title("📊 Patron Management Tool")
st.markdown("Upload, filter, sort, and export your data easily.")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    data = load_data(uploaded_file)
else:
    st.info("Using default file: summary_Patrons.xlsx")
    data = load_data(open("summary_Patrons.xlsx", "rb"))

if data.empty:
    st.warning("No data to display. Upload a valid file.")
else:
    # Filter Sidebar
    st.sidebar.header("Filter Options")
    filtered_data = data.copy()

    for col in data.columns:
        unique_vals = data[col].dropna().unique()
        if len(unique_vals) <= 20:
            selected_vals = st.sidebar.multiselect(f"{col}", unique_vals)
            if selected_vals:
                filtered_data = filtered_data[filtered_data[col].isin(selected_vals)]

    # Sorting
    st.sidebar.header("Sort Data")
    sort_col = st.sidebar.selectbox("Sort by column", filtered_data.columns)
    sort_order = st.sidebar.radio("Sort Order", ["Ascending", "Descending"]) == "Ascending"
    filtered_data = filtered_data.sort_values(by=sort_col, ascending=sort_order)

    # Limit rows
    top_n = st.sidebar.number_input("Number of rows to display (leave blank for all)", min_value=1, value=len(filtered_data), step=1)
    filtered_data = filtered_data.head(top_n)

    # Output filename
    st.sidebar.header("Export Data")
    output_filename = st.sidebar.text_input("Output filename", value="filtered_data.csv")

    # Display Data
    st.write("### Filtered and Sorted Data")
    st.dataframe(filtered_data)

    # Export button
    st.download_button(
        label="📥 Download data as CSV",
        data=filtered_data.to_csv(index=False).encode('utf-8'),
        file_name=output_filename,
        mime='text/csv'
    )
"""

import pandas as pd
import streamlit as st

# Function to load data
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

# Streamlit UI
st.title("Patron Management Tool")

# Default file path
default_file = "summary_Patrons.xlsx"  # Replace with your default file path

# File uploader
#uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Load the default file if no file is uploaded
#if uploaded_file:
#    data = load_data(uploaded_file)
#else:
#    st.write(f"No file uploaded. Using default file: {default_file}")
#    data = load_data(default_file)

st.write(f"No file uploaded. Using default file: {default_file}")
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

# Top N filter
top_n = st.sidebar.number_input("Top N rows (leave blank for all)", min_value=1, step=1, value=len(data))

if sort_column:
    data = data.sort_values(by=sort_column, ascending=ascending).head(top_n)

# Output file name selection
st.sidebar.title("Output Options")
default_output_file = "filtered_data.csv"
output_file_name = st.sidebar.text_input("Enter output file name:", default_output_file)

# Display filtered/sorted data
st.write("### Filtered/Sorted Data", data)

# Download button
st.download_button(
    "Download Filtered Data",
    data.to_csv(index=False),
    file_name=output_file_name,
    mime="text/csv"
)
"""