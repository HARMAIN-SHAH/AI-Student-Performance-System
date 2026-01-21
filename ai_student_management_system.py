# AI-Based Student Performance & prediction and analytics system with  secure teacher authentication.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import hashlib
import os

# =============================================
# 1️⃣ ADVANCE TEACHER LOGIN
# =============================================
def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

def teacher_login():
    teachers = pd.read_csv("teachers.csv")
    attempts = 3

    while attempts > 0:
        print("\n=== Secure Teacher Login ===")
        username = input("Username: ")
        password = input("Password: ")

        password_hash = hash_password(password)

        user = teachers[
            (teachers['username'] == username) &
            (teachers['password_hash'] == password_hash)
        ]

        if not user.empty:
            print("✅ Login Successful! Access Granted.\n")
            return True
        else:
            attempts -= 1
            print(f"❌ Invalid credentials. Attempts left: {attempts}")

    print("🚫 Too many failed attempts. Access denied.")
    return False

if not teacher_login():
    exit()

# =============================================
# 2️⃣ DATASET
# =============================================
data = {
    "Name": ["Ali","Sara","Ahmed","Zara","Usman","Aisha","Bilal","Hina","Fahad","Maya","Omar","Sana"],
    "Attendance": [90,85,70,60,50,95,88,45,78,92,55,40],
    "MidMarks": [88,80,65,50,45,92,78,40,70,90,48,30],
    "FinalMarks": [90,85,60,55,42,95,80,38,75,92,50,35],
    "Assignments": [92,88,70,60,45,96,85,40,80,94,55,38],
    "Result": [1,1,1,0,0,1,1,0,1,1,0,0]
}

df = pd.DataFrame(data)

# =============================================
# 3️⃣ DATA PREPROCESSING
# =============================================
X = df[["Attendance","MidMarks","FinalMarks","Assignments"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =============================================
# 4️⃣ AI MODELS TRAINING
# =============================================
dt_model = DecisionTreeClassifier(criterion="entropy")
dt_model.fit(X_train, y_train)
acc_dt = accuracy_score(y_test, dt_model.predict(X_test))

rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, rf_model.predict(X_test))

best_model = dt_model if acc_dt >= acc_rf else rf_model
best_model_name = "Decision Tree" if acc_dt >= acc_rf else "Random Forest"
best_accuracy = max(acc_dt, acc_rf)

# =============================================
# 5️⃣ SINGLE STUDENT PREDICTION (FIXED)
# =============================================
def predict_student():
    print("\n--- Predict Single Student Performance ---")
    name = input("Student Name: ")
    att = int(input("Attendance (%): "))
    mid = int(input("Mid Exam Marks: "))
    final = int(input("Final Exam Marks: "))
    assign = int(input("Assignment Score: "))

    student = pd.DataFrame(
        [[att, mid, final, assign]],
        columns=["Attendance","MidMarks","FinalMarks","Assignments"]
    )

    result = best_model.predict(student)[0]

    avg = (att + mid + final + assign) / 4
    if avg >= 85:
        level = "Excellent"
    elif avg >= 70:
        level = "Good"
    elif avg >= 50:
        level = "Average"
    else:
        level = "Poor"

    rec = "Encourage advanced learning activities" if result == 1 else "Extra classes, counseling & monitoring required"

    print(f"\nStudent: {name}")
    print(f"Prediction: {'PASS' if result==1 else 'AT RISK'}")
    print(f"Performance Level: {level}")
    print(f"Recommendation: {rec}")

    report = pd.DataFrame(
        [[name, att, mid, final, assign, result, level, rec]],
        columns=["Name","Attendance","MidMarks","FinalMarks","Assignments","Prediction","Level","Recommendation"]
    )

    if os.path.exists("predictions.csv"):
        report.to_csv("predictions.csv", mode="a", header=False, index=False)
    else:
        report.to_csv("predictions.csv", index=False)

    print("✅ Prediction saved to predictions.csv")

# =============================================
# 6️⃣ BATCH PREDICTION 
# =============================================
def batch_prediction():
    import os
    import pandas as pd
    import numpy as np

    print("\n--- Batch Prediction Mode ---")
    file = input("Enter CSV filename: ").strip()

    if not os.path.exists(file):
        print("❌ File not found. Make sure file is in same folder as Python script.")
        return

    # Read CSV
    batch = pd.read_csv(file)
    batch.columns = [c.strip() for c in batch.columns]  # remove extra spaces

    required_cols = ["Attendance", "MidMarks", "FinalMarks", "Assignments"]
    for col in required_cols:
        if col not in batch.columns:
            print(f"❌ Missing column: {col}")
            print("Available columns:", list(batch.columns))
            return

    # Ensure numeric
    for col in required_cols:
        batch[col] = pd.to_numeric(batch[col], errors='coerce')
    batch = batch.dropna(subset=required_cols)

    results = []

    for i, row in batch.iterrows():
        name = row["Name"] if "Name" in batch.columns else f"Student_{i+1}"

        student_df = pd.DataFrame([[
            row["Attendance"],
            row["MidMarks"],
            row["FinalMarks"],
            row["Assignments"]
        ]], columns=["Attendance","MidMarks","FinalMarks","Assignments"])

        # Predict using best_model (DecisionTree or RandomForest)
        prediction = best_model.predict(student_df)[0]

        # Average score for performance level
        avg = (row["Attendance"] + row["MidMarks"] + row["FinalMarks"] + row["Assignments"]) / 4
        if avg >= 85:
            level = "Excellent"
        elif avg >= 70:
            level = "Good"
        elif avg >= 50:
            level = "Average"
        else:
            level = "Poor"

        rec = "Encourage advanced learning activities" if prediction==1 else "Extra classes & counseling required"

        results.append([name, row["Attendance"], row["MidMarks"], row["FinalMarks"], row["Assignments"], prediction, level, rec])

    # Create DataFrame
    df_out = pd.DataFrame(results, columns=[
        "Name","Attendance","MidMarks","FinalMarks","Assignments","Prediction","Level","Recommendation"
    ])

    # ✅ Save to batch_predictions.csv (OVERWRITE mode)
    df_out.to_csv("batch_predictions.csv", index=False)

    print("✅ All students predictions SAVED in batch_predictions.csv (OVERWRITTEN)\n")

    # ✅ Show table in terminal
    print("📄 Latest Batch Predictions Preview:\n")
    print(df_out.to_string(index=False))  # nice table without row numbers


# =============================================
# 7️⃣ GRAPHS & INSIGHTS
# =============================================
def show_insights():
    print("\n📊 Statistical Insights")
    print("Average Attendance:", round(df["Attendance"].mean(),2))
    print("Average Final Marks:", round(df["FinalMarks"].mean(),2))
    print("At Risk Students:", (df["Result"]==0).sum())

    plt.bar(df["Name"], df["FinalMarks"])
    plt.title("Final Marks of Students")
    plt.xlabel("Students")
    plt.ylabel("Marks")
    plt.show()

    plt.pie(
        [df["Result"].sum(), len(df)-df["Result"].sum()],
        labels=["PASS","AT RISK"],
        autopct="%1.1f%%"
    )
    plt.title("PASS vs AT RISK")
    plt.show()

def advanced_graphs(df_graph):
    import matplotlib.pyplot as plt
    
    # Average Marks
    df_graph["Average"] = (df_graph["MidMarks"] + df_graph["FinalMarks"] + df_graph["Assignments"]) / 3

    # 1️⃣ Attendance vs Final Marks
    plt.figure()
    plt.scatter(df_graph["Attendance"], df_graph["FinalMarks"])
    plt.xlabel("Attendance")
    plt.ylabel("Final Marks")
    plt.title("Attendance vs Final Marks")
    plt.show()

    # 2️⃣ Average Marks Distribution
    plt.figure()
    plt.hist(df_graph["Average"], bins=5)
    plt.xlabel("Average Marks")
    plt.ylabel("Number of Students")
    plt.title("Average Marks Distribution")
    plt.show()

    # 3️⃣ PASS vs AT RISK (Bar Chart)
    plt.figure()
    df_graph["Prediction"].value_counts().plot(kind="bar")
    plt.xlabel("Result (1=PASS, 0=AT RISK)")
    plt.ylabel("Students Count")
    plt.title("PASS vs AT RISK Students")
    plt.show()

    # 4️⃣ Subject-wise Performance
    subjects = ["MidMarks", "FinalMarks", "Assignments"]
    averages = [df_graph[s].mean() for s in subjects]

    plt.figure()
    plt.bar(subjects, averages)
    plt.xlabel("Assessment Type")
    plt.ylabel("Average Marks")
    plt.title("Subject-wise Performance Analysis")
    plt.show()

    # 5️⃣ Top vs Bottom Students
    sorted_df = df_graph.sort_values("Average", ascending=False)

    top5 = sorted_df.head(5)
    bottom5 = sorted_df.tail(5)

    plt.figure()
    plt.barh(top5["Name"], top5["Average"])
    plt.title("Top 5 Students (Average Marks)")
    plt.xlabel("Average Marks")
    plt.show()

    plt.figure()
    plt.barh(bottom5["Name"], bottom5["Average"])
    plt.title("Bottom 5 Students (Average Marks)")
    plt.xlabel("Average Marks")
    plt.show()


# =============================================
# 8️⃣ DATASET VIEW
# =============================================
def view_dataset():
    print("\n📁 Dataset:\n")
    print(df)

# =============================================
# 9️⃣ MODEL REPORT
# =============================================
def view_model_report():
    print(f"\nBest Model: {best_model_name}")
    print(f"Accuracy: {best_accuracy*100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, best_model.predict(X_test)))

# =============================================
# 🔟 FUTURE SCOPE
# =============================================
def future_scope():
    print("\n🚀 Future Scope")
    print("- Web dashboard (Flask / Streamlit)")
    print("- Deep learning integration")
    print("- Real-time monitoring")
    print("- Personalized AI recommendations")

# =============================================
# 11️⃣ MAIN MENU
# =============================================
while True:
    print("\n===============================")
    print(" AI Student Performance System ")
    print("===============================")
    print("1. Predict Single Student")
    print("2. Batch Prediction")
    print("3. View Dataset")
    print("4. Model Accuracy & Report")
    print("5. Graphs & Insights")
    print("6. Future Scope")
    print("7. Advanced Graphs")
    print("8. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        predict_student()
    elif choice == "2":
        batch_prediction()
    elif choice == "3":
        view_dataset()
    elif choice == "4":
        view_model_report()
    elif choice == "5":
        show_insights()
    elif choice == "6":
        future_scope()
    elif choice == "7":
        # ✅ FIXED: read batch_predictions.csv before calling graph function
        try:
            df_batch = pd.read_csv("batch_predictions.csv")
            advanced_graphs(df_batch)
        except FileNotFoundError:
            print("❌ batch_predictions.csv not found. Pehle batch prediction run karein.")
    elif choice=="8":
        print("\nSystem Closed Successfully. Thank You!")
        break
    else:
        print("❌ Invalid Choice. Try Again.")
