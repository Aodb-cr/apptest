import pandas as pd
import numpy as np
import io
#pip install streamlit
import streamlit as st
#pip install openpyxl
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
import matplotlib.pyplot as plt
#pip install statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot

#Random_data 
from sklearn.metrics import mean_squared_error, r2_score
import math

#Modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

#pip install scikit-optimize
from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real


def Home():
    st.header("TEST!!!")
    st.title('Estimate Remaining Useful Life of Fire-Tube Boilers')
    image_url = 'https://static.thairath.co.th/media/dFQROr7oWzulq5Fa5nLx41LBjuIcpaDPqzIWb455g4S14Wzkx0LoiwnF6UHonFMb86L.jpg'
    st.image(image_url, use_column_width=True)
    st.write("""
        One of the main challenges in industrial operations is to reduce maintenance costs and minimize downtime, all while maintaining or improving safety standards. A significant portion of operational delays is due to unplanned maintenance—such as when a boiler shows abnormal behavior, causing disruptions that can result in costly downtime or even system shutdowns.

        Since it is difficult to know when a boiler will fail, maintenance schedules tend to be conservative, especially for safety-critical equipment like Fire-Tube Boilers. However, scheduling maintenance too early wastes the remaining useful life of the boiler and its components, leading to increased operational costs.

        If we can predict the remaining useful life (RUL) of a Fire-Tube Boiler, maintenance can be scheduled just before a failure occurs. Implementing predictive maintenance minimizes unnecessary maintenance, reduces production hours lost to downtime, and lowers the cost of spare parts by optimizing their usage.""")
    st.subheader('Data for Predictive Maintenance of Fire-Tube Boilers')
    st.write('''
        In this case, we're utilizing sensor data from Fire-Tube Boilers, which includes a variety of measurements such as temperature, pressure, flow rates, and other operational parameters. The data also includes information about different operational conditions and fault modes.

        The data typically comprises several cycles of the boiler's life under various operating conditions. During these cycles, the boiler operates normally at the start and gradually develops a fault until failure occurs. In a test set, the time series ends sometime before the failure point.''')
    st.subheader('Problem Statement')
    st.write('''
        The goal is to predict, at any time t, using the available historical and current data, when the Fire-Tube Boiler is likely to fail in the near future.

        The problem can be approached in two ways:

        1. Classification: Predicting the probability that the boiler will fail within a pre-specified time window.
        2. Regression: Estimating the remaining time to failure, or the remaining useful life (RUL) of the Fire-Tube Boiler.''')



# ฟังก์ชันหลัก
def main():
    # สร้าง sidebar header
    st.sidebar.header("Navigation")

    # เมนูตัวเลือกใน sidebar โดยตั้งค่าเริ่มต้นเป็นหน้า 'Home'
    page = st.sidebar.selectbox("Go to", ['Home', "Data & Edit File", "Visualizations",'Features' ,'Model', "Predict"], index=0)

    # วิดเจ็ตสำหรับการอัปโหลดไฟล์
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        
        # ตรวจสอบชนิดไฟล์
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        
        # แสดงผลหน้าตามที่ผู้ใช้เลือกจากเมนูใน sidebar
        if page == "Data & Edit File":
            Edit_file(df)
        elif page == "Visualizations":
            Visualizations(df)
        elif page == "Model":
            st.header('!!!Modeling!!!')
            target_column = st.selectbox('Select the target column for prediction:', df.columns)
            model(df,target_column)
        elif page == "Predict":
            predict(df)
        elif page == 'Home':
            Home()
        elif page == 'Features':
            Features(df)

    else:
        # ถ้าไม่มีการอัปโหลดไฟล์ ให้แสดงหน้า Home
        if page == 'Home':
            Home()
        else:
            st.warning("Please upload a file to proceed.")

def Edit_file(df):
    st.header("Data & Edit File")
    st.subheader("Data Overview")

    # เพิ่มตัวเลือกในการดูข้อมูลในไฟล์ CSV ทั้งหมด
    st.write("Data in CSV File:")
    st.dataframe(df)

    # วาง "Number of rows and columns" และ "Column Names" ไว้ข้างกัน
    col2, col1  = st.columns(2)
    
    with col1:
        st.markdown("### Number of rows and columns")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
    
    with col2:
        st.markdown("### Column Names:")
        st.write(df.columns)

    st.subheader("Descriptive Statistics")
    st.write(df.describe(include='all'))

    # สร้างคอลัมน์ข้างกัน
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Values")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])

    with col2:
        st.subheader("Data Types")
        st.write(df.dtypes)

    # ลบข้อมูลที่หายไป (missing data)
    if st.checkbox("Remove Missing Data (drop rows with NaN)"):
        df_cleaned = df.dropna()
        st.write("Data after removing missing values:")
        st.dataframe(df_cleaned)
    else:
        df_cleaned = df
    
    # วิดเจ็ตสำหรับการแก้ไขข้อมูล
    row_index = st.number_input("Select Row to Edit", min_value=0, max_value=len(df_cleaned)-1)
    column_to_edit = st.selectbox("Select Column to Edit", df_cleaned.columns)
    
    # รับค่าข้อมูลใหม่จากผู้ใช้
    new_value = st.text_input("Enter New Value", value=str(df_cleaned.loc[row_index, column_to_edit]))
    
    # ปุ่มบันทึกการแก้ไข
    if st.button("Save Changes"):
        df_cleaned.loc[row_index, column_to_edit] = new_value
        st.success(f"Updated row {row_index}, column {column_to_edit} with new value: {new_value}")
    
    # แสดงผลข้อมูลที่แก้ไขแล้ว
    if st.checkbox("View Updated Data"):
        st.write("Updated Data:")
        st.dataframe(df_cleaned)
    
    # ฟังก์ชันสำหรับการดาวน์โหลดไฟล์ CSV ที่แก้ไขแล้ว
    download_csv(df_cleaned)

# ฟังก์ชันสำหรับการดาวน์โหลดไฟล์ CSV
def download_csv(df):
    # แปลง DataFrame ให้เป็นไฟล์ CSV ในหน่วยความจำ
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    # ปุ่มสำหรับการดาวน์โหลดไฟล์ CSV
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="edited_data.csv",
        mime="text/csv"
    )

def Visualizations(df):
    st.header("Visualizations")
    
    # ตรวจสอบข้อมูล missing
    if df.isnull().values.any():
        st.warning("The dataset contains missing values. Please go back to 'Edit File' to handle them.")
        if st.button("Go to Edit File"):
            st.session_state.page = "Data & Edit File"  # ใช้สถานะเซสชันเพื่อเปลี่ยนหน้าต่าง ๆ
            return

    if 'Tanggal' in df.columns:
        if df['Tanggal'].dtype == object:
            try:
                df['Tanggal'] = pd.to_datetime(df['Tanggal'])
            except:
                st.error("Column 'Tanggal' cannot be converted to datetime.")
                return
    
        df.set_index('Tanggal', inplace=True)
    
        has_time = df.index.to_series().dt.time.notnull().any()

        # ตั้งค่าฟอร์แมตเวลาที่ผู้ใช้เลือก
        time_format_option = st.sidebar.selectbox(
            "Select Time Format",
            ["YYYY-MM-DD", "YYYY-MM-DD HH:MM:SS", "YYYY-MM-DD HH:MM", "Custom"]
        )

        time_format = {
            "YYYY-MM-DD": "%Y-%m-%d",
            "YYYY-MM-DD HH:MM:SS": "%Y-%m-%d %H:%M:%S",
            "YYYY-MM-DD HH:MM": "%Y-%m-%d %H:%M"
        }.get(time_format_option, time_format_option)  # Use the custom format if provided

        min_date = df.index.min().date()
        max_date = df.index.max().date()

        start_date = st.sidebar.date_input("Start date", min_date)
        end_date = st.sidebar.date_input("End date", max_date)

        if has_time:
            min_time = df.index.to_series().dt.time.min() if not df.empty else pd.to_datetime('00:00:00').time()
            max_time = df.index.to_series().dt.time.max() if not df.empty else pd.to_datetime('23:59:59').time()

            start_time = st.sidebar.time_input("Start time", min_time)
            end_time = st.sidebar.time_input("End time", max_time)

            df_filtered = df[
                (df.index.date >= start_date) &
                (df.index.date <= end_date) &
                (df.index.to_series().dt.time.fillna(pd.to_datetime('00:00:00').time()) >= start_time) &
                (df.index.to_series().dt.time.fillna(pd.to_datetime('23:59:59').time()) <= end_time)
            ]
        else:
            df_filtered = df[
                (df.index.date >= start_date) &
                (df.index.date <= end_date)
            ]

        # ใช้ฟอร์แมตเวลาที่เลือกเพื่อแสดงผล
        df_filtered.index = df_filtered.index.strftime(time_format)

        st.write(f"Data from {start_date} {start_time if has_time else ''} to {end_date} {end_time if has_time else ''}")
        st.dataframe(df_filtered)

        selected_columns = st.sidebar.multiselect("Select columns to plot", df_filtered.columns)

        if len(selected_columns) > 0:
            # Visualization options
            #show_decomposition = st.sidebar.checkbox("Show Time Series Decomposition")
            show_rolling = st.sidebar.checkbox("Show Rolling Statistics")
            show_autocorrelation = st.sidebar.checkbox("Show Autocorrelation Plot")

            # # Time Series Decomposition
            # if show_decomposition and 'Tanggal' in df.columns and pd.api.types.is_datetime64_any_dtype(df.index):
            #     st.subheader('Time Series Decomposition')
            #     try:
            #         # Ensure the data is sorted by date
            #         df_sorted = df.sort_index()
            #         series = df_sorted[selected_columns[0]]
            #         result = seasonal_decompose(series, model='additive', period=365)  # Adjust period as needed
                    
            #         st.write("Trend Component")
            #         plt.figure(figsize=(12, 6))
            #         plt.plot(result.trend.dropna())
            #         plt.title(f'Trend Component of {selected_columns[0]}')
            #         plt.xlabel('Tanggal (Date)')
            #         plt.ylabel('Value')
            #         x_labels = plt.gca().get_xticks()
            #         if len(x_labels) > 10:
            #             interval = max(1, len(x_labels) // 10)  # ป้องกันการหารด้วยศูนย์
            #             plt.gca().set_xticks(x_labels[::interval])
            #         plt.xticks(rotation=45)
            #         st.pyplot(plt)
                     
            #         st.write("Seasonal Component")
            #         plt.figure(figsize=(12, 6))
            #         plt.plot(result.seasonal.dropna())
            #         plt.title(f'Seasonal Component of {selected_columns[0]}')
            #         plt.xlabel('Tanggal (Date)')
            #         plt.ylabel('Value')
            #         x_labels = plt.gca().get_xticks()
            #         if len(x_labels) > 10:
            #             interval = max(1, len(x_labels) // 10)
            #             plt.gca().set_xticks(x_labels[::interval])
            #         plt.xticks(rotation=45)
            #         st.pyplot(plt)

            #         st.write("Residual Component")
            #         plt.figure(figsize=(12, 6))
            #         plt.plot(result.resid.dropna())
            #         plt.title(f'Residual Component of {selected_columns[0]}')
            #         plt.xlabel('Tanggal (Date)')
            #         plt.ylabel('Value')
            #         x_labels = plt.gca().get_xticks()
            #         if len(x_labels) > 10:
            #             interval = max(1, len(x_labels) // 10)
            #             plt.gca().set_xticks(x_labels[::interval])
            #         plt.xticks(rotation=45)
            #         st.pyplot(plt)
            #     except Exception as e:
            #         st.error(f"Time Series Decomposition failed: {e}")

            # Rolling Statistics
            if show_rolling:
                rolling_window = st.sidebar.slider("Rolling Window Size", min_value=1, max_value=365, value=30)
                st.subheader('Rolling Statistics')
                for column in selected_columns:
                    st.write(f"Rolling Mean for {column} with window size {rolling_window}")
                    rolling_mean = df_filtered[column].rolling(window=rolling_window).mean()
                    plt.figure(figsize=(12, 6))
                    plt.plot(df_filtered.index, df_filtered[column], label=f'{column} Actual')
                    plt.plot(df_filtered.index, rolling_mean, label=f'{column} Rolling Mean', color='red')
                    plt.xlabel('Tanggal (Date and Time)' if has_time else 'Tanggal (Date)')
                    plt.ylabel(column)
                    plt.title(f'Rolling Mean of {column}')
                    plt.legend()
                    x_labels = plt.gca().get_xticks()
                    if len(x_labels) > 10:
                        interval = max(1, len(x_labels) // 10)
                        plt.gca().set_xticks(x_labels[::interval])
                    plt.xticks(rotation=45)
                    st.pyplot(plt)
                    plt.close()

            # Autocorrelation Plot
            if show_autocorrelation:
                st.subheader('Autocorrelation')
                for column in selected_columns:
                    plt.figure(figsize=(12, 6))
                    autocorrelation_plot(df_filtered[column])
                    plt.title(f'Autocorrelation of {column}')
                    x_labels = plt.gca().get_xticks()
                    if len(x_labels) > 10:
                        interval = max(1, len(x_labels) // 10)
                        plt.gca().set_xticks(x_labels[::interval])
                    plt.xticks(rotation=45)
                    st.pyplot(plt)
                    plt.close()
        else:
            st.write("Please select at least one column to plot.")

def Features(df):
    # st.header("Features Overview")

    # st.subheader("Data Correlation Matrix")
    # st.write("This shows the correlation between different features in the dataset.")
    
    # # แสดงตาราง Correlation
    # correlation_matrix = df.corr()
    # st.write(correlation_matrix)
    
    # # เลือกฟีเจอร์ที่ต้องการดูข้อมูล
    # selected_features = st.multiselect("Select Features to Explore", df.columns)
    
    # if selected_features:
    #     st.subheader(f"Selected Features: {selected_features}")
    #     st.write(df[selected_features].describe())
        
    #     # แสดงข้อมูล Correlation เฉพาะฟีเจอร์ที่เลือก
    #     st.subheader("Correlation of Selected Features")
    #     st.write(df[selected_features].corr())

    # # เพิ่มการแสดงผลความสำคัญของฟีเจอร์ (Feature Importance) สำหรับโมเดลแบบง่าย เช่น RandomForest
    # if st.checkbox("Show Feature Importance (using RandomForest)"):
    #     from sklearn.ensemble import RandomForestRegressor

    #     y = df.pop(df.columns[0])  # สมมติให้คอลัมน์แรกเป็น target (สามารถแก้ไขได้ตามต้องการ)
        
    #     # สร้างโมเดล RandomForest
    #     model = RandomForestRegressor()
    #     model.fit(df, y)

    #     # ดึงความสำคัญของฟีเจอร์
    #     feature_importance = pd.Series(model.feature_importances_, index=df.columns).sort_values(ascending=False)
        
    #     st.write("Feature Importance:")
    #     st.bar_chart(feature_importance)

    # # ปิดท้ายด้วยการแสดงข้อมูลบางส่วนของฟีเจอร์
    # st.subheader("Sample Data:")
    # st.write(df.head())
    pass


# def model(data):
#     pass

def model(data, selected_target):

    
    # Features and target
    features = [
        'Main steam temperature (boiler side) (℃)', 
        'Main steam pressure (boiler side) (Mpa)',
        'Reheat steam temperature (boiler side) (℃)',
        'Feedwater temperature (℃)', 
        'Feedwater flow (t/h)',
        'Flue gas temperature (℃)', 
        'Boiler oxygen level (%)', 
        'Coal Flow (t/h)'
    ]
    target = selected_target

    # Step 1: Split the data
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Train the model
    rf_model = LinearRegression()
    rf_model.fit(X_train, y_train)

    # Step 3: Evaluate the model
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    r2 = r2_score(y_test, y_pred_test)

    # Display model performance metrics
    st.write(f"Train RMSE: {train_rmse}")
    st.write(f"Test RMSE: {test_rmse}")
    st.write(f"R² Score: {r2}")

    # Step 4: Display Correlation Matrix
    st.write("Correlation Matrix:")
    corr_matrix = data[features].corr()
    pltcorr, axcorr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axcorr)
    st.pyplot(pltcorr)

    # Step 5: Optimization using Scipy
    def predict_efficiency(params):
        params = np.array(params).reshape(1, -1)
        return -rf_model.predict(params)[0]  # Negative because we want to maximize efficiency

    initial_guess = [565, 14.5, 565, 230, 840, 125, 4.5, 170]
    bounds = [
        (550, 580),  # Main steam temperature
        (13, 16),    # Main steam pressure
        (550, 580),  # Reheat steam temperature
        (220, 240),  # Feedwater temperature
        (800, 900),  # Feedwater flow
        (100, 150),  # Flue gas temperature
        (3, 5),      # Boiler oxygen level
        (150, 200)   # Coal flow
    ]

    result = minimize(predict_efficiency, initial_guess, bounds=bounds, method='L-BFGS-B')
    optimized_params = result.x
    predicted_efficiency = -result.fun

    st.write("Optimized Parameters (Scipy):")
    for i, param in enumerate(optimized_params):
        st.write(f"{features[i]}: {param}")
    st.write(f"Predicted maximum boiler efficiency: {predicted_efficiency}%")

    # Step 6: Bayesian Optimization
    space = [
        Real(550, 580, name='Main steam temperature (℃)'),
        Real(13, 16, name='Main steam pressure (Mpa)'),
        Real(550, 580, name='Reheat steam temperature (℃)'),
        Real(220, 240, name='Feedwater temperature (℃)'),
        Real(800, 900, name='Feedwater flow (t/h)'),
        Real(100, 150, name='Flue gas temperature (℃)'),
        Real(3, 5, name='Boiler oxygen level (%)'),
        Real(150, 200, name='Coal flow (t/h)')
    ]

    res_bo = gp_minimize(predict_efficiency, space, n_calls=50, random_state=42)
    optimized_params_bo = res_bo.x
    predicted_efficiency_bo = -res_bo.fun

    st.write("Optimized Parameters from Bayesian Optimization:")
    for i, param in enumerate(optimized_params_bo):
        st.write(f"{features[i]}: {param}")
    st.write(f"Predicted maximum boiler efficiency with Bayesian Optimization: {predicted_efficiency_bo}%")

# # Streamlit UI
# st.title("Boiler Efficiency Prediction and Optimization")
# uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
#     st.write("Dataset Loaded Successfully")
    
#     # Display a selectbox for choosing the target column
#     target_column = st.selectbox('Select the target column for prediction:', data.columns)
    
#     # Run the model with the selected target
#     model(data, target_column)
# else:
#     st.write("Please upload a CSV file to proceed.")



def predict(df):
    st.header("!!!Data_Randomize!!!")
    features = [
        'Main steam temperature (boiler side) (℃)',
        'Main steam pressure (boiler side) (Mpa)',
        'Reheat steam temperature (boiler side) (℃)',
        'Feedwater temperature (℃)',
        'Feedwater flow (t/h)',
        'Flue gas temperature (℃)',
        'Boiler oxygen level (%)',
        'Circulating water outlet temperature (℃)',
        'NTHR (Kcal/Kwh)', 
        'NPHR (Kcal/Kwh)', 
        'Gross Load (MW)',
        'Nett Load (MW)', 
        'Energy Input From Boiler (Kcal/h)', 
        'Coal Flow (t/h)', 
        'Boiler Eff (%)'
    ]

    # Step 1: Ensure the necessary columns exist
    available_features = [col for col in features if col in df.columns]
    missing_features = [col for col in features if col not in df.columns]
    
    if missing_features:
        st.warning(f"The following columns are missing: {missing_features}")

    df = df[available_features]

    # Step 2: Correlation matrix calculation
    correlation_matrix = df.corr()

    if 'Boiler Eff (%)' in df.columns:
        corr_matrix = correlation_matrix['Boiler Eff (%)']
    else:
        st.warning("'Boiler Eff (%)' not in DataFrame. Skipping correlation calculations.")
        return

    # Step 3: Perform Cholesky decomposition (check if correlation matrix is positive definite)
    try:
        cholesky_decomp = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        st.error("Correlation matrix is not positive definite. Cholesky decomposition failed.")
        return

    # Step 4: Generate random normal data
    days_simulated = 90
    random_data = np.random.normal(size=(days_simulated, len(df.columns)))

    # Step 5: Adjust the random data using Cholesky decomposition to maintain correlations
    adjusted_data = random_data @ cholesky_decomp.T
    adjusted_data = pd.DataFrame(adjusted_data, columns=df.columns)

    # Step 6: Scale the data back to the original means and standard deviations
    for col in df.columns:
        adjusted_data[col] = adjusted_data[col] * df[col].std() + df[col].mean()

    # Step 7: Add random noise
    random_noise = np.random.normal(0, 1, size=(days_simulated, len(df.columns)))

    # Step 8: Adjust the noise using Cholesky decomposition to maintain correlations
    adjusted_noise = random_noise @ cholesky_decomp.T

    # Step 9: Scale the noise back to original standard deviations
    scaled_noise = pd.DataFrame(adjusted_noise, columns=df.columns)
    for col in df.columns:
        scaled_noise[col] = scaled_noise[col] * df[col].std()

    # Step 10: Add noise to the adjusted data
    noisy_data = adjusted_data + scaled_noise

    # Output the generated noisy data
    st.write("Generated Noisy Data:")
    st.write(noisy_data)

    # Save the noisy data to a CSV file
    noisy_data.to_csv(f"Random_data_{days_simulated}.csv", index=False)

    # Load the data back for plotting
    randomize_data = pd.read_csv(f"Random_data_{days_simulated}.csv")

    # Plot the simulated future data
    fig, ax = plt.subplots(figsize=(15, 9))
    for col in randomize_data.columns:
        if col not in ['Energy Input From Boiler (Kcal/h)', 'Unnamed: 0']:
            ax.plot(randomize_data[col], label=f'Simulated {col}')
    ax.legend()
    ax.set_title(f'Simulated Future Data (Next {days_simulated} Days)')
    ax.set_xlabel('Days')
    ax.set_ylabel('Value')
    ax.grid(True)
    st.pyplot(fig)  # Display the plot in Streamlit

    # Show statistics of the simulated data
    st.write("Simulated Data Statistics:")
    st.write(randomize_data.describe())

    # Plot the original data for comparison
    fig2, ax2 = plt.subplots(figsize=(15, 9))
    for col in df.columns:
        if col not in ['Energy Input From Boiler (Kcal/h)', 'Unnamed: 0']:
            ax2.plot(df[col], label=f'Original {col}')
    ax2.legend()
    ax2.set_title('Original Data')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    st.pyplot(fig2)  # Display the original data plot in Streamlit

    # Show statistics of the original data
    st.write("Original Data Statistics:")
    st.write(df.describe())


if __name__ == "__main__":
    main()
