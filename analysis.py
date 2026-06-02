import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join("data", "titanic.csv")
OUTPUT_DIR = "visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================
# Load Dataset
# ==========================


def load_dataset(path):
    try:
        df = pd.read_csv(path)
        print("Dataset Loaded Successfully")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")
        return df

    except FileNotFoundError:
        print(f"Dataset not found at: {path}")
        raise


# ==========================
# Basic Analysis
# ==========================


def basic_analysis(df):

    print("\n========== DATA INFO ==========")
    df.info()

    print("\n========== STATISTICS ==========")
    print(df.describe(include="all"))

    avg_age = df["Age"].mean()
    avg_fare = df["Fare"].mean()
    survival_rate = df["Survived"].mean() * 100

    print(f"\nAverage Age: {avg_age:.2f}")
    print(f"Average Fare: {avg_fare:.2f}")
    print(f"Overall Survival Rate: {survival_rate:.2f}%")

    return avg_age, avg_fare, survival_rate


# ==========================
# Missing Values Handling
# ==========================


def clean_data(df):

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Cabin" in df.columns:
        df.drop(columns=["Cabin"], inplace=True)

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    print("\n========== AFTER CLEANING ==========")
    print(df.isnull().sum())

    return df


# ==========================
# Bar Chart
# ==========================


def create_bar_chart(df):
    survival_by_class = df.groupby("Pclass")["Survived"].mean() * 100

    ax = survival_by_class.plot(
        kind="bar", color=["#2ecc71", "#3498db", "#e74c3c"], edgecolor="black"
    )

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2, p.get_height() + 2),
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate (%)")
    plt.xticks(rotation=0)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.savefig(os.path.join(OUTPUT_DIR, "bar_chart.png"), dpi=300, bbox_inches="tight")

    plt.close()


# ==========================
# Scatter Plot
# ==========================


def create_scatter_plot(df):
    plt.figure(figsize=(10, 6))

    # Fix: Use 'No' and 'Yes' directly as keys so the legend maps perfectly automatically
    sns.scatterplot(
        data=df,
        x="Age",
        y="Fare",
        hue=df["Survived"].map(
            {0: "No", 1: "Yes"}
        ),  # Dynamic mapping safely handles the legend
        palette={"No": "red", "Yes": "green"},  # Clean semantic mapping
        alpha=0.7,
    )

    plt.title(
        "Age vs Fare (Colored by Survival Status)", fontsize=14, fontweight="bold"
    )

    plt.xlabel("Age (Years)")
    plt.ylabel("Fare (Ticket Price)")

    # Clean up legend title without hardcoding position-sensitive labels
    plt.legend(title="Survived Status")

    plt.grid(True, linestyle="--", alpha=0.5)

    # Ensuring OUTPUT_DIR exists dynamically inside the function to avoid FileNotFoundError
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.savefig(
        os.path.join(OUTPUT_DIR, "scatter_plot.png"), dpi=300, bbox_inches="tight"
    )

    plt.close()


# ==========================
# Heatmap
# ==========================


def create_heatmap(df):

    df["Sex_encoded"] = df["Sex"].map({"female": 0, "male": 1})

    columns = ["Survived", "Pclass", "Age", "Fare", "SibSp", "Parch", "Sex_encoded"]

    corr = df[columns].corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar=True,
    )

    plt.title("Correlation Heatmap")

    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap.png"), dpi=300, bbox_inches="tight")

    plt.close()


# ==========================
# Histogram
# ==========================


def create_histogram(df):

    plt.figure(figsize=(8, 5))

    sns.histplot(df["Age"], bins=20, kde=True)

    plt.title("Age Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Age (Years)")
    plt.ylabel("Number of Passengers")

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.savefig(
        os.path.join(OUTPUT_DIR, "age_histogram.png"), dpi=300, bbox_inches="tight"
    )

    plt.close()


def create_gender_survival_chart(df):

    plt.figure(figsize=(6, 5))

    survival_by_gender = df.groupby("Sex")["Survived"].mean() * 100

    survival_by_gender.plot(kind="bar", color=["purple", "orange"])

    plt.title("Survival Rate by Gender")

    plt.ylabel("Survival Rate (%)")

    plt.xticks(rotation=0)

    plt.savefig(
        os.path.join(OUTPUT_DIR, "gender_survival.png"), dpi=300, bbox_inches="tight"
    )

    plt.close()


# ==========================
# Insights Generation
# ==========================


def generate_insights(df):

    female_survival = df[df["Sex"] == "female"]["Survived"].mean() * 100

    male_survival = df[df["Sex"] == "male"]["Survived"].mean() * 100

    insights = []

    insights.append(f"Female Survival Rate: {female_survival:.2f}%")

    insights.append(f"Male Survival Rate: {male_survival:.2f}%")

    if female_survival > male_survival:
        insights.append("Women had a significantly higher survival rate than men.")

    survival_by_class = df.groupby("Pclass")["Survived"].mean() * 100

    best_class = survival_by_class.idxmax()

    insights.append(f"Passenger Class {best_class} had the highest survival rate.")

    youngest = df["Age"].min()
    oldest = df["Age"].max()
    insights.append(f"Age ranged from {youngest:.0f} to {oldest:.0f} years.")

    highest_fare = df["Fare"].max()

    insights.append(f"Highest ticket fare was {highest_fare:.2f}.")
    insights.append(f"Average ticket fare was {df['Fare'].mean():.2f}.")
    corr_survival_fare = df["Survived"].corr(df["Fare"])

    insights.append(
        f"Correlation between Fare and Survival was {corr_survival_fare:.2f}."
    )
    with open("insights.txt", "w", encoding="utf-8") as file:

        file.write("\n".join(insights))

    print("\n========== INSIGHTS ==========")

    for item in insights:
        print(item)

    return female_survival, male_survival


# ==========================
# Export Summary
# ==========================


def export_summary(
    df, avg_age, avg_fare, survival_rate, male_survival, female_survival
):

    summary = {
        "Passengers": len(df),
        "Average Age": round(avg_age, 2),
        "Average Fare": round(avg_fare, 2),
        "Survival Rate (%)": round(survival_rate, 2),
        "Male Survival Rate (%)": round(male_survival, 2),
        "Female Survival Rate (%)": round(female_survival, 2),
    }

    pd.DataFrame([summary]).to_csv("summary.csv", index=False)

    print("\nsummary.csv created successfully")


# ==========================
# Main Function
# ==========================


def main():

    df = load_dataset(DATA_PATH)

    df = clean_data(df)

    avg_age, avg_fare, survival_rate = basic_analysis(df)

    create_bar_chart(df)
    create_scatter_plot(df)
    create_heatmap(df)
    create_histogram(df)
    create_gender_survival_chart(df)
    female_survival, male_survival = generate_insights(df)

    export_summary(df, avg_age, avg_fare, survival_rate, male_survival, female_survival)

    print("\nProject Completed Successfully")


if __name__ == "__main__":
    main()
