import numpy as np
import pandas as pd

# ==========================================
# 1. Load and inspect the app usage dataset
# ==========================================
file_path = "24_mobile_app_usage.csv"
df = pd.read_csv(file_path)

print("=== 1. DATASET INSPECTION ===")
print("Shape (rows, columns):", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary statistics:\n", df.describe(include="all"))


# ==========================================
# 2. Check for missing values and duplicate Session_ID records
# ==========================================
print("\n=== 2. MISSING VALUES & DUPLICATES ===")
print("Missing values per column:\n", df.isnull().sum())

duplicate_sessions = df.duplicated(subset=["Session_ID"]).sum()
print(f"Duplicate Session_ID count: {duplicate_sessions}")

# Remove exact duplicate Session_ID rows if any exist
df = df.drop_duplicates(subset=["Session_ID"]).copy()
print("Shape after removing duplicate Session_IDs:", df.shape)


# ==========================================
# 3. Handle missing Session_Duration_Mins or User_Rating with a justified approach
# ==========================================
# Justification:
# - Session_Duration_Mins is often skewed (due to long sessions), so imputing missing
#   values with the MEDIAN is more robust than using the mean.
# - User_Rating is an ordinal/bounded score, so imputing missing values with the
#   MEDIAN (or rounded mean) preserves central tendency without distortion from outliers.
print("\n=== 3. HANDLING MISSING VALUES ===")
df["Session_Duration_Mins"] = pd.to_numeric(df["Session_Duration_Mins"], errors="coerce")
df["User_Rating"] = pd.to_numeric(df["User_Rating"], errors="coerce")

duration_median = df["Session_Duration_Mins"].median()
rating_median = df["User_Rating"].median()

df["Session_Duration_Mins"] = df["Session_Duration_Mins"].fillna(duration_median)
df["User_Rating"] = df["User_Rating"].fillna(rating_median)

print(f"Filled missing Session_Duration_Mins with median: {duration_median:.2f}")
print(f"Filled missing User_Rating with median: {rating_median:.2f}")
print("Remaining nulls in key columns:", df[["Session_Duration_Mins", "User_Rating"]].isnull().sum().to_dict())


# ==========================================
# 4. Convert Session_Date to datetime and extract month, weekday, and hour
# ==========================================
print("\n=== 4. DATETIME EXTRACTION ===")
df["Session_Date"] = pd.to_datetime(df["Session_Date"], errors="coerce")
df["Month"] = df["Session_Date"].dt.month
df["Month_Name"] = df["Session_Date"].dt.month_name()
df["Weekday"] = df["Session_Date"].dt.day_name()
df["Hour"] = df["Session_Date"].dt.hour

print(df[["Session_Date", "Month", "Month_Name", "Weekday", "Hour"]].head())


# ==========================================
# 5. Calculate total sessions, unique users, and average session duration
# ==========================================
print("\n=== 5. CORE ENGAGEMENT METRICS ===")
total_sessions = len(df)
unique_users_count = df["User_ID"].nunique()
avg_session_duration = df["Session_Duration_Mins"].mean()

print(f"Total Sessions: {total_sessions}")
print(f"Unique Users: {unique_users_count}")
print(f"Average Session Duration (mins): {avg_session_duration:.2f}")


# ==========================================
# 6. Find the 10 most frequently used app features
# ==========================================
print("\n=== 6. TOP 10 MOST FREQUENTLY USED FEATURES ===")
top_10_features = df["Feature_Used"].value_counts().head(10)
print(top_10_features)


# ==========================================
# 7. Compare average session duration across device types
# ==========================================
print("\n=== 7. AVERAGE SESSION DURATION BY DEVICE TYPE ===")
device_duration = df.groupby("Device_Type")["Session_Duration_Mins"].agg(
    session_count="count",
    avg_duration_mins="mean",
    median_duration_mins="median"
).sort_values(by="avg_duration_mins", ascending=False)
print(device_duration)


# ==========================================
# 8. Rank user age groups by total number of sessions
# ==========================================
print("\n=== 8. AGE GROUPS RANKED BY TOTAL SESSIONS ===")
age_group_sessions = (
    df.groupby("Age_Group")["Session_ID"]
    .count()
    .reset_index(name="Total_Sessions")
    .sort_values(by="Total_Sessions", ascending=False)
)
age_group_sessions["Rank"] = range(1, len(age_group_sessions) + 1)
print(age_group_sessions[["Rank", "Age_Group", "Total_Sessions"]])


# ==========================================
# 9. Compare average user ratings across app features
# ==========================================
print("\n=== 9. AVERAGE USER RATINGS ACROSS APP FEATURES ===")
feature_ratings = (
    df.groupby("Feature_Used")["User_Rating"]
    .agg(avg_rating="mean", rating_count="count")
    .sort_values(by="avg_rating", ascending=False)
)
print(feature_ratings)


# ==========================================
# 10. Identify peak app usage hours and the busiest weekdays
# ==========================================
print("\n=== 10. PEAK HOURS AND BUSIEST WEEKDAYS ===")
hourly_usage = df["Hour"].value_counts().sort_values(ascending=False)
weekday_usage = df["Weekday"].value_counts()

print("Top 5 Peak Usage Hours (0-23):\n", hourly_usage.head(5))
print("\nSessions by Weekday (Busiest first):\n", weekday_usage)


# ==========================================
# 11. Use a list to store most-used features and a tuple to store fixed device categories
# ==========================================
print("\n=== 11. LIST & TUPLE DATA STRUCTURES ===")
most_used_features_list = top_10_features.index.tolist()
fixed_device_categories_tuple = tuple(sorted(df["Device_Type"].dropna().unique()))

print("Most-used features (list):", most_used_features_list)
print("Fixed device categories (tuple):", fixed_device_categories_tuple)


# ==========================================
# 12. Use a set to identify unique users & write a function for avg duration of a selected feature
# ==========================================
print("\n=== 12. SET OF UNIQUE USERS & FEATURE DURATION FUNCTION ===")
unique_users_set = set(df["User_ID"].dropna())
print("Unique users count via set:", len(unique_users_set))


def avg_duration_for_feature(data: pd.DataFrame, feature_name: str) -> float:
    """Return the average session duration (in minutes) for a selected app feature."""
    filtered = data[data["Feature_Used"] == feature_name]
    if filtered.empty:
        raise ValueError(f"Feature '{feature_name}' not found in dataset.")
    return float(filtered["Session_Duration_Mins"].mean())


sample_feature = most_used_features_list[0]
print(f"Average duration for '{sample_feature}': {avg_duration_for_feature(df, sample_feature):.2f} mins")


# ==========================================
# 13. Use NumPy to calculate mean, median, and standard deviation of session duration
# ==========================================
print("\n=== 13. NUMPY SESSION DURATION STATISTICS ===")
durations_array = df["Session_Duration_Mins"].to_numpy()
np_mean = np.mean(durations_array)
np_median = np.median(durations_array)
np_std = np.std(durations_array, ddof=1)  # sample standard deviation

print(f"Mean Duration (NumPy): {np_mean:.2f} mins")
print(f"Median Duration (NumPy): {np_median:.2f} mins")
print(f"Std Deviation (NumPy): {np_std:.2f} mins")


# ==========================================
# 14. Identify unusually long sessions using IQR & create engagement labels (Low, Medium, High)
# ==========================================
print("\n=== 14. IQR OUTLIERS & ENGAGEMENT LABELS ===")
q1 = np.percentile(durations_array, 25)
q3 = np.percentile(durations_array, 75)
iqr = q3 - q1
upper_outlier_threshold = q3 + 1.5 * iqr
lower_outlier_threshold = q1 - 1.5 * iqr

df["Is_Long_Session_Outlier"] = df["Session_Duration_Mins"] > upper_outlier_threshold
outlier_count = df["Is_Long_Session_Outlier"].sum()

print(f"Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
print(f"Upper Outlier Threshold (Q3 + 1.5*IQR): {upper_outlier_threshold:.2f} mins")
print(f"Unusually long session outliers found: {outlier_count}")


# Documented Engagement Rules:
# - High: Session_Duration_Mins >= Q3 OR User_Rating >= 4.5
# - Medium: Session_Duration_Mins >= Q1 OR User_Rating >= 3.0
# - Low: Below Medium thresholds
def assign_engagement_label(row: pd.Series) -> str:
    duration = row["Session_Duration_Mins"]
    rating = row["User_Rating"]
    if duration >= q3 or rating >= 4.5:
        return "High"
    elif duration >= q1 or rating >= 3.0:
        return "Medium"
    else:
        return "Low"


df["Engagement_Level"] = df.apply(assign_engagement_label, axis=1)
print("\nEngagement Level Distribution:\n", df["Engagement_Level"].value_counts())

# Save cleaned dataset for deliverables
cleaned_output_file = "cleaned_mobile_app_usage.csv"
df.to_csv(cleaned_output_file, index=False)
print(f"\nCleaned dataset saved to '{cleaned_output_file}'.")


# ==========================================
# 15. Create an AppUser class & write five recommendations to improve engagement
# ==========================================
print("\n=== 15. APPUSER CLASS & RECOMMENDATIONS ===")


class AppUser:
    """Represents a mobile app user with user ID and device type."""

    def __init__(self, user_id, device_type: str):
        self.user_id = user_id
        self.device_type = device_type

    def summary(self) -> str:
        return f"AppUser(user_id={self.user_id}, device_type='{self.device_type}')"


# Example instantiation using first row
first_row = df.iloc[0]
sample_user = AppUser(user_id=first_row["User_ID"], device_type=first_row["Device_Type"])
print("Sample instance:", sample_user.summary())

recommendations = [
    "1. Optimize top-performing features identified in the Top 10 list by reducing load times and surfacing quick shortcuts on the home screen.",
    "2. Improve lower-rated features by collecting in-app micro-feedback after use and fixing usability bottlenecks.",
    "3. Schedule push notifications and feature campaigns around peak usage hours and busiest weekdays to boost retention.",
    "4. Tailor onboarding and content recommendations for the highest-session age groups while testing targeted prompts for under-engaged segments.",
    "5. Ensure consistent UI performance and smooth session continuity across all device categories in the device tuple."
]

print("\nFive Recommendations to Improve App Engagement:")
for rec in recommendations:
    print(rec)
import pandas as pd

# ==========================================
# 1. Load and inspect the app usage dataset
# ==========================================
file_path = "24_mobile_app_usage.csv"
df = pd.read_csv(file_path)

print("=== 1. DATASET INSPECTION ===")
print("Shape (rows, columns):", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary statistics:\n", df.describe(include="all"))


# ==========================================
# 2. Check for missing values and duplicate Session_ID records
# ==========================================
print("\n=== 2. MISSING VALUES & DUPLICATES ===")
print("Missing values per column:\n", df.isnull().sum())

duplicate_sessions = df.duplicated(subset=["Session_ID"]).sum()
print(f"Duplicate Session_ID count: {duplicate_sessions}")

# Remove exact duplicate Session_ID rows if any exist
df = df.drop_duplicates(subset=["Session_ID"]).copy()
print("Shape after removing duplicate Session_IDs:", df.shape)


# ==========================================
# 3. Handle missing Session_Duration_Mins or User_Rating with a justified approach
# ==========================================
# Justification:
# - Session_Duration_Mins is often skewed (due to long sessions), so imputing missing
#   values with the MEDIAN is more robust than using the mean.
# - User_Rating is an ordinal/bounded score, so imputing missing values with the
#   MEDIAN (or rounded mean) preserves central tendency without distortion from outliers.
print("\n=== 3. HANDLING MISSING VALUES ===")
df["Session_Duration_Mins"] = pd.to_numeric(df["Session_Duration_Mins"], errors="coerce")
df["User_Rating"] = pd.to_numeric(df["User_Rating"], errors="coerce")

duration_median = df["Session_Duration_Mins"].median()
rating_median = df["User_Rating"].median()

df["Session_Duration_Mins"] = df["Session_Duration_Mins"].fillna(duration_median)
df["User_Rating"] = df["User_Rating"].fillna(rating_median)

print(f"Filled missing Session_Duration_Mins with median: {duration_median:.2f}")
print(f"Filled missing User_Rating with median: {rating_median:.2f}")
print("Remaining nulls in key columns:", df[["Session_Duration_Mins", "User_Rating"]].isnull().sum().to_dict())


# ==========================================
# 4. Convert Session_Date to datetime and extract month, weekday, and hour
# ==========================================
print("\n=== 4. DATETIME EXTRACTION ===")
df["Session_Date"] = pd.to_datetime(df["Session_Date"], errors="coerce")
df["Month"] = df["Session_Date"].dt.month
df["Month_Name"] = df["Session_Date"].dt.month_name()
df["Weekday"] = df["Session_Date"].dt.day_name()
df["Hour"] = df["Session_Date"].dt.hour

print(df[["Session_Date", "Month", "Month_Name", "Weekday", "Hour"]].head())


# ==========================================
# 5. Calculate total sessions, unique users, and average session duration
# ==========================================
print("\n=== 5. CORE ENGAGEMENT METRICS ===")
total_sessions = len(df)
unique_users_count = df["User_ID"].nunique()
avg_session_duration = df["Session_Duration_Mins"].mean()

print(f"Total Sessions: {total_sessions}")
print(f"Unique Users: {unique_users_count}")
print(f"Average Session Duration (mins): {avg_session_duration:.2f}")


# ==========================================
# 6. Find the 10 most frequently used app features
# ==========================================
print("\n=== 6. TOP 10 MOST FREQUENTLY USED FEATURES ===")
top_10_features = df["Feature_Used"].value_counts().head(10)
print(top_10_features)


# ==========================================
# 7. Compare average session duration across device types
# ==========================================
print("\n=== 7. AVERAGE SESSION DURATION BY DEVICE TYPE ===")
device_duration = df.groupby("Device_Type")["Session_Duration_Mins"].agg(
    session_count="count",
    avg_duration_mins="mean",
    median_duration_mins="median"
).sort_values(by="avg_duration_mins", ascending=False)
print(device_duration)


# ==========================================
# 8. Rank user age groups by total number of sessions
# ==========================================
print("\n=== 8. AGE GROUPS RANKED BY TOTAL SESSIONS ===")
age_group_sessions = (
    df.groupby("Age_Group")["Session_ID"]
    .count()
    .reset_index(name="Total_Sessions")
    .sort_values(by="Total_Sessions", ascending=False)
)
age_group_sessions["Rank"] = range(1, len(age_group_sessions) + 1)
print(age_group_sessions[["Rank", "Age_Group", "Total_Sessions"]])


# ==========================================
# 9. Compare average user ratings across app features
# ==========================================
print("\n=== 9. AVERAGE USER RATINGS ACROSS APP FEATURES ===")
feature_ratings = (
    df.groupby("Feature_Used")["User_Rating"]
    .agg(avg_rating="mean", rating_count="count")
    .sort_values(by="avg_rating", ascending=False)
)
print(feature_ratings)


# ==========================================
# 10. Identify peak app usage hours and the busiest weekdays
# ==========================================
print("\n=== 10. PEAK HOURS AND BUSIEST WEEKDAYS ===")
hourly_usage = df["Hour"].value_counts().sort_values(ascending=False)
weekday_usage = df["Weekday"].value_counts()

print("Top 5 Peak Usage Hours (0-23):\n", hourly_usage.head(5))
print("\nSessions by Weekday (Busiest first):\n", weekday_usage)


# ==========================================
# 11. Use a list to store most-used features and a tuple to store fixed device categories
# ==========================================
print("\n=== 11. LIST & TUPLE DATA STRUCTURES ===")
most_used_features_list = top_10_features.index.tolist()
fixed_device_categories_tuple = tuple(sorted(df["Device_Type"].dropna().unique()))

print("Most-used features (list):", most_used_features_list)
print("Fixed device categories (tuple):", fixed_device_categories_tuple)


# ==========================================
# 12. Use a set to identify unique users & write a function for avg duration of a selected feature
# ==========================================
print("\n=== 12. SET OF UNIQUE USERS & FEATURE DURATION FUNCTION ===")
unique_users_set = set(df["User_ID"].dropna())
print("Unique users count via set:", len(unique_users_set))


def avg_duration_for_feature(data: pd.DataFrame, feature_name: str) -> float:
    """Return the average session duration (in minutes) for a selected app feature."""
    filtered = data[data["Feature_Used"] == feature_name]
    if filtered.empty:
        raise ValueError(f"Feature '{feature_name}' not found in dataset.")
    return float(filtered["Session_Duration_Mins"].mean())


sample_feature = most_used_features_list[0]
print(f"Average duration for '{sample_feature}': {avg_duration_for_feature(df, sample_feature):.2f} mins")


# ==========================================
# 13. Use NumPy to calculate mean, median, and standard deviation of session duration
# ==========================================
print("\n=== 13. NUMPY SESSION DURATION STATISTICS ===")
durations_array = df["Session_Duration_Mins"].to_numpy()
np_mean = np.mean(durations_array)
np_median = np.median(durations_array)
np_std = np.std(durations_array, ddof=1)  # sample standard deviation

print(f"Mean Duration (NumPy): {np_mean:.2f} mins")
print(f"Median Duration (NumPy): {np_median:.2f} mins")
print(f"Std Deviation (NumPy): {np_std:.2f} mins")


# ==========================================
# 14. Identify unusually long sessions using IQR & create engagement labels (Low, Medium, High)
# ==========================================
print("\n=== 14. IQR OUTLIERS & ENGAGEMENT LABELS ===")
q1 = np.percentile(durations_array, 25)
q3 = np.percentile(durations_array, 75)
iqr = q3 - q1
upper_outlier_threshold = q3 + 1.5 * iqr
lower_outlier_threshold = q1 - 1.5 * iqr

df["Is_Long_Session_Outlier"] = df["Session_Duration_Mins"] > upper_outlier_threshold
outlier_count = df["Is_Long_Session_Outlier"].sum()

print(f"Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
print(f"Upper Outlier Threshold (Q3 + 1.5*IQR): {upper_outlier_threshold:.2f} mins")
print(f"Unusually long session outliers found: {outlier_count}")


# Documented Engagement Rules:
# - High: Session_Duration_Mins >= Q3 OR User_Rating >= 4.5
# - Medium: Session_Duration_Mins >= Q1 OR User_Rating >= 3.0
# - Low: Below Medium thresholds
def assign_engagement_label(row: pd.Series) -> str:
    duration = row["Session_Duration_Mins"]
    rating = row["User_Rating"]
    if duration >= q3 or rating >= 4.5:
        return "High"
    elif duration >= q1 or rating >= 3.0:
        return "Medium"
    else:
        return "Low"


df["Engagement_Level"] = df.apply(assign_engagement_label, axis=1)
print("\nEngagement Level Distribution:\n", df["Engagement_Level"].value_counts())

# Save cleaned dataset for deliverables
cleaned_output_file = "cleaned_mobile_app_usage.csv"
df.to_csv(cleaned_output_file, index=False)
print(f"\nCleaned dataset saved to '{cleaned_output_file}'.")


# ==========================================
# 15. Create an AppUser class & write five recommendations to improve engagement
# ==========================================
print("\n=== 15. APPUSER CLASS & RECOMMENDATIONS ===")


class AppUser:
    """Represents a mobile app user with user ID and device type."""

    def __init__(self, user_id, device_type: str):
        self.user_id = user_id
        self.device_type = device_type

    def summary(self) -> str:
        return f"AppUser(user_id={self.user_id}, device_type='{self.device_type}')"


# Example instantiation using first row
first_row = df.iloc[0]
sample_user = AppUser(user_id=first_row["User_ID"], device_type=first_row["Device_Type"])
print("Sample instance:", sample_user.summary())

recommendations = [
    "1. Optimize top-performing features identified in the Top 10 list by reducing load times and surfacing quick shortcuts on the home screen.",
    "2. Improve lower-rated features by collecting in-app micro-feedback after use and fixing usability bottlenecks.",
    "3. Schedule push notifications and feature campaigns around peak usage hours and busiest weekdays to boost retention.",
    "4. Tailor onboarding and content recommendations for the highest-session age groups while testing targeted prompts for under-engaged segments.",
    "5. Ensure consistent UI performance and smooth session continuity across all device categories in the device tuple."
]

print("\nFive Recommendations to Improve App Engagement:")
for rec in recommendations:
    print(rec)