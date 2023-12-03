import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Function to create containers in the sidebar
def create_sidebar_container(subheader):
    container = st.sidebar.container()
    container.subheader(subheader)
    return container

# Function to simulate data loading
def load_data():
    # Add radio buttons for choosing data source
    data_source = st.sidebar.radio("Choose Data Source", ["Sample Data", "Upload CSV"])

    if data_source == "Sample Data":
        df = pd.DataFrame({
            'Column1': np.random.rand(100),
            'Column2': np.random.randint(1, 100, size=100),
            'Column3': np.random.choice(['A', 'B', 'C'], size=100)
        })

        # 1. Importing and Data Collection Container
        with create_sidebar_container('1. Importing and Data Collection'):
            st.write("Using Sample Data.")
            # st.write(df)

    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # 1. Importing and Data Collection Container
            with create_sidebar_container('1. Importing and Data Collection'):
                st.write("Using Uploaded CSV Data.")
                # st.write(df)
    # else:
    #     df = pd.DataFrame()  # Placeholder for other cases

    return df

# Function for data inspection
import streamlit as st
import numpy as np

def data_inspection(df):
    # 2. Data Inspection Container
    with st.sidebar:
        st.subheader('2. Data Inspection')
        # Checkbox for df.head()
        show_head = st.checkbox("Show df.head()")
        # Checkbox for df.tail()
        show_tail = st.checkbox("Show df.tail()")
        # Checkbox for df.info()
        show_info = st.checkbox("Show df.info()")
        # Checkbox for df.shape
        show_shape = st.checkbox("Show df.shape")
        # Checkbox for df.dtypes
        show_dtypes = st.checkbox("Show df.dtypes")
        # Checkbox for df.count()
        show_count = st.checkbox("Show df.count()")
        # Checkbox for df.describe()
        show_describe = st.checkbox("Show df.describe()")
        # Checkbox for df.columns
        show_columns = st.checkbox("Show df.columns")
        # Checkbox for df.duplicated().sum()
        show_duplicates = st.checkbox("Show df.duplicated().sum()")
        # Checkbox for df.isnull().sum()
        show_nulls = st.checkbox("Show df.isnull().sum()")
        # Checkbox for IQR calculation
        show_iqr = st.checkbox("Show IQR for Numeric Columns")
        
    if show_head:
            st.write("### DataFrame Head")
            st.write(df.head())
    if show_tail:
            st.write("### DataFrame Tail")
            st.write(df.tail())
    if show_info:
            st.write("### DataFrame Info")
            st.write(df.info())
    if show_shape:
            st.write("### DataFrame Shape")
            st.write(df.shape)
    if show_dtypes:
            st.write("### DataFrame Data Types")
            st.write(df.dtypes)
    if show_count:
            st.write("### DataFrame Count")
            st.write(df.count())
    if show_describe:
            st.write("### DataFrame Describe")
            st.write(df.describe())
    if show_columns:
            st.write("### DataFrame Columns")
            st.write(df.columns)
    if show_duplicates:
            st.write("### Duplicates in DataFrame")
            st.write(df.duplicated().sum())
    if show_nulls:
            st.write("### Null Values in DataFrame")
            st.write(df.isnull().sum())
    if show_iqr:
            st.write("### IQR for Numeric Columns")
            # Assuming df_numeric is a subset of numeric columns from df
            df_numeric = df.select_dtypes(include=np.number)
            for column in df_numeric.columns:
                Q1, Q3 = df_numeric[column].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                st.write(f"IQR for {column}: {IQR}")

    return df

# Function to handle duplicates
# Function for handling duplicates
def handle_duplicates(df):
    # 3. Duplicates Handling Container
    with create_sidebar_container('3. Duplicates Handling'):
        # Checkbox for dropping duplicates
        drop_duplicates = st.checkbox("Drop Duplicate Rows")
    # Print data shape and count
    st.write("### Original Data:")
    st.write("Shape:", df.shape)
    st.write("Count:")
    st.write(df.count())

    # Detect duplicates
    duplicates_df = df[df.duplicated()]

    # Print duplicates shape and count
    st.write("\n### Duplicates:")
    st.write("Shape:", duplicates_df.shape)
    st.write("Count:")
    st.write(duplicates_df.count())
    if drop_duplicates:
            # Drop duplicates
            df = df.drop_duplicates()

            # Print data shape and count after dropping duplicates
            st.write("\n### Data after dropping duplicates:")
            st.write("Shape:", df.shape)
            st.write("Count:")
            st.write(df.count())

    return df


# Function for handling null values
def handle_null_values(df):
    # 4. Handling Null Values Container
    with create_sidebar_container('4. Handling Null Values'):
        # Show null value count for each column
        st.write("### Null Values Count for Each Column:")
        st.write(df.isnull().sum())

        # Radio button to choose between drop and impute
        null_handling_option = st.radio("Select Null Handling Option:", ["Drop Null", "Impute Null"])

        if null_handling_option == "Drop Null":
            # Checkbox to select columns for drop/impute (numerical and categorical)
            selected_columns = st.multiselect("Select Columns to Drop Null Values:", df.columns)

            if selected_columns:
                # Drop null values
                df = df.dropna(subset=selected_columns)

                # Show null count after dropping null values
                st.write("\n### Data after Dropping Null Values:")
                st.write("Shape:", df.shape)
                st.write("Count:")
                st.write(df.isnull().sum())

        elif null_handling_option == "Impute Null":
            # Checkbox to select columns for drop/impute (numerical and categorical)
            selected_columns = st.multiselect("Select Columns to Impute Null Values:", df.columns)

            if selected_columns:
                # Imputation method selection
                imputation_method = st.selectbox("Select Imputation Method:", ["Mean", "Median", "Mode"])

                # Impute null values based on selected method
                if imputation_method == "Mean":
                    df[selected_columns] = df[selected_columns].fillna(df[selected_columns].mean())
                elif imputation_method == "Median":
                    df[selected_columns] = df[selected_columns].fillna(df[selected_columns].median())
                elif imputation_method == "Mode":
                    df[selected_columns] = df[selected_columns].fillna(df[selected_columns].mode().iloc[0])

                # Show null count after imputing null values
                st.write("\n### Data after Imputing Null Values:")
                st.write("Shape:", df.shape)
                st.write("Count:")
                st.write(df.isnull().sum())
    

    return df


# Function to handle outliers
def handle_outliers(df):
    # 5. Handling Outliers Container
    with create_sidebar_container('5. Handling Outliers'):
        # Your code for handling outliers goes here
        pass
    return df

# Function for correlation analysis
def correlation_analysis(df):
    # 6. Correlation Analysis Container
    with create_sidebar_container('6. Correlation Analysis'):
        # Your code for correlation analysis goes here
        pass
    return df

# Function for data visualization
def visualize_data(df):
    # 7. Data Visualization Container
    with create_sidebar_container('7. Data Visualization'):
        # Your code for data visualization goes here
        st.write('Here I will visualize')
    return df

# Function for data transformation
def data_transformation(df):
    # 8. Data Transformation Container
    with create_sidebar_container('8. Data Transformation'):
        # Your code for data transformation goes here
        pass
    return df

# Main function
def main():
    st.title("Data Analytics Web App")

    # Data Analytics Container
    with st.container():
        st.header("Data Analytics")
    
    # Load data and call other functions
    df = load_data()
    data_inspection(df)
    handle_duplicates(df)
    handle_null_values(df)
    handle_outliers(df)
    correlation_analysis(df)
    visualize_data(df)
    data_transformation(df)

if __name__ == "__main__":
    main()
