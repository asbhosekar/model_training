import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import median_absolute_error, PredictionErrorDisplay

#DO PIP INSTALLS - pip install streamlit pandas matplotlib seaborn numpy scipy scikit-learn

# ---------------------------
# INITIAL SETUP
# ---------------------------
st.title("Data Analysis and Model Training App")

# Session state for step tracking
if "step" not in st.session_state:
    st.session_state.step = 0

# ---------------------------
# FILE UPLOAD (Optional)
# ---------------------------
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("Basic Information of the DataFrame:")
    st.dataframe(df.describe())
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

# ---------------------------
# LOAD SAMPLE DATA
# ---------------------------
wages = fetch_openml(data_id=534, as_frame=True)
df_wages = wages.data
target_wage = wages.target

st.dataframe(df_wages.head())
st.dataframe(target_wage.head())

# ---------------------------
# STEP 0 → STEP 1
# ---------------------------
if st.session_state.step == 0:
    if st.button("YES - Proceed to Data Analysis", key="step0"):
        st.session_state.step = 1
    else:
        st.info("Click YES to proceed with 1. Visualize numerical feature distributions.")
        st.stop()

# ---------------------------
# STEP 1: Numerical Feature Distributions
# ---------------------------
if st.session_state.step >= 1:
    numerical_cols = df_wages.select_dtypes(include=['int64', 'float64']).columns
    st.header("1. Visualize numerical feature distributions")
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        plt.hist(df_wages[col], bins=20)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        st.pyplot(plt)

    if st.session_state.step == 1:
        if st.button("YES - Proceed to Categorical Features", key="step1"):
            st.session_state.step = 2
        else:
            st.info("Click YES to proceed to 2. Categorical feature distributions.")
            st.stop()

# ---------------------------
# STEP 2: Categorical Feature Distributions
# ---------------------------
if st.session_state.step >= 2:
    categorical_cols = df_wages.select_dtypes(include=['object', 'category']).columns
    st.header("2. Visualize categorical feature distributions")
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df_wages, x=col)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

    if st.session_state.step == 2:
        if st.button("YES - Proceed to Numerical Relationships", key="step2"):
            st.session_state.step = 3
        else:
            st.info("Click YES to proceed to 3. relationships with numerical target.")
            st.stop()

# ---------------------------
# STEP 3: Relationships with Numerical Target
# ---------------------------
if st.session_state.step >= 3:
    st.header("3. Visualize relationships with the NUMERICAL variables")
    numerical_cols = df_wages.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_wages, x=col, y=target_wage)
        plt.title(f'Relationship between {col} and WAGE')
        plt.xlabel(col)
        plt.ylabel('WAGE')
        st.pyplot(plt)

    if st.session_state.step == 3:
        if st.button("YES - Proceed to Categorical Relationships", key="step3"):
            st.session_state.step = 4
        else:
            st.info("Click YES to proceed to 4. Relationships with categorical target.")
            st.stop()

# ---------------------------
# STEP 4: Relationships with Categorical Target
# ---------------------------
if st.session_state.step >= 4:
    st.header("4. Visualize relationships with the target CATEGORICAL variables")
    categorical_cols = df_wages.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_wages, x=col, y=target_wage)
        plt.title(f'Relationship between {col} and WAGE')
        plt.xlabel(col)
        plt.ylabel('WAGE')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt)

    if st.session_state.step == 4:
        if st.button("YES - Proceed to Model Training & Pairplot", key="step4"):
            st.session_state.step = 5
        else:
            st.info("Click YES to proceed to 5. Model training and pairwise plots.")
            st.stop()

# ---------------------------
# STEP 5: Model Training & Pairplot
# ---------------------------
if st.session_state.step >= 5:
    st.header("5. Display Model Training & Pairwise relationships of numerical features")
    X = df_wages
    y = target_wage
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    X_train_numerical = X_train.select_dtypes(include=['int64', 'float64'])
    train_dataset = X_train_numerical.copy()
    train_dataset.insert(0, "WAGE", y_train)
    fig = sns.pairplot(train_dataset, kind="reg", diag_kind="kde")
    plt.suptitle("Pairwise relationships of numerical features in the training set", y=1.02)
    st.pyplot(fig.fig)
    plt.close(fig.fig)

    if st.session_state.step == 5:
        if st.button("YES - Proceed to Prediction Error Plot", key="step5"):
            st.session_state.step = 6
        else:
            st.info("Click YES to proceed to 6. Prediction error plot.")
            st.stop()

# ---------------------------
# STEP 6: Prediction Error Plot
# ---------------------------
if st.session_state.step >= 6:
    st.header("6. Display the prediction error plot")
    categorical_columns = ["RACE", "OCCUPATION", "SECTOR", "MARR", "UNION", "SEX", "SOUTH"]
    numerical_columns = ["EDUCATION", "EXPERIENCE", "AGE"]

    preprocessor = make_column_transformer(
        (OneHotEncoder(drop="if_binary"), categorical_columns),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    model = make_pipeline(
        preprocessor,
        TransformedTargetRegressor(
            regressor=Ridge(alpha=1e-10), func=np.log10, inverse_func=sp.special.exp10
        ),
    )
    model.fit(X_train, y_train)

    mae_train = median_absolute_error(y_train, model.predict(X_train))
    y_pred = model.predict(X_test)
    mae_test = median_absolute_error(y_test, y_pred)
    scores = {
        "MedAE on training set": f"{mae_train:.2f} $/hour",
        "MedAE on testing set": f"{mae_test:.2f} $/hour",
    }
    st.write(scores)

    _, ax = plt.subplots(figsize=(5, 5))
    PredictionErrorDisplay.from_predictions(
        y_test, y_pred, kind="actual_vs_predicted", ax=ax, scatter_kwargs={"alpha": 0.5}
    )
    ax.set_title("Ridge model, small regularization")
    for name, score in scores.items():
        ax.plot([], [], " ", label=f"{name}: {score}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(plt)

    if st.session_state.step == 6:
        if st.button("YES - Proceed to Model Coefficients", key="step6"):
            st.session_state.step = 7
        else:
            st.info("Click YES to proceed to 7. Model coefficients.")
            st.stop()

# ---------------------------
# STEP 7: Model Coefficients
# ---------------------------
if st.session_state.step >= 7:
    st.header("7. MODEL COEFFICIENTS")
    feature_names = model[:-1].get_feature_names_out()
    coefs = pd.DataFrame(
        model[-1].regressor_.coef_,
        columns=["Coefficients"],
        index=feature_names,
    )
    st.write(coefs)

    # Trigger to move from Step 7 to Step 8
    if st.session_state.step == 7:
        if st.button("YES - Proceed to Coefficient Bar Plot", key="step7"):
            st.session_state.step = 8
        else:
            st.info("Click YES to proceed to coefficient bar plot.")
            st.stop()


# ---------------------------
# STEP 8: Coefficient Bar Plot
# ---------------------------
if st.session_state.step >= 8:
    st.header("8. Visualize Model Coefficients as Horizontal Bar Chart")

    fig, ax = plt.subplots(figsize=(9, 7))
    coefs.plot.barh(ax=ax)
    ax.set_title("Ridge model, small regularization")
    ax.axvline(x=0, color=".5")
    ax.set_xlabel("Raw coefficient values")
    plt.subplots_adjust(left=0.3)
    st.pyplot(fig)

# Trigger to move from Step 8 to Step 9
if st.session_state.step == 8:
    if st.button("YES - Proceed to Feature Ranges", key="step8"):
        st.session_state.step = 9
    else:
        st.info("Click YES to proceed to 9. Feature Ranges.")
        st.stop()


# ---------------------------
# STEP 9: Feature Ranges (Std. Dev. of Preprocessed Features)
# ---------------------------
if st.session_state.step >= 9:
    st.header("9. Feature Ranges (Standard Deviation after Preprocessing)")

    # Transform training data using the preprocessing pipeline (excluding the final regressor)
    X_train_preprocessed = pd.DataFrame(
        model[:-1].transform(X_train),
        columns=feature_names
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    X_train_preprocessed.std(axis=0).plot.barh(ax=ax)
    ax.set_title("Feature ranges")
    ax.set_xlabel("Std. dev. of feature values")
    plt.subplots_adjust(left=0.3)
    st.pyplot(fig)

# Trigger to move from Step 9 to Step 10
if st.session_state.step == 9:
    if st.button("YES - Proceed to Coeficient Importance", key="step9"):
        st.session_state.step = 10
    else:
        st.info("Click YES to proceed to 10. Coefficient Importance (Corrected by Feature Std. Dev.)")
        st.stop()



# ---------------------------
# STEP 10: Coefficient Importance (Corrected by Feature Std. Dev.)
# ---------------------------
if st.session_state.step >= 10:
    st.header("10. Coefficient Importance (Corrected by Feature Std. Dev.)")

    # Multiply coefficients by the std. dev. of the corresponding preprocessed feature
    coefs_corrected = pd.DataFrame(
        model[-1].regressor_.coef_ * X_train_preprocessed.std(axis=0),
        columns=["Coefficient importance"],
        index=feature_names,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    coefs_corrected.plot(kind="barh", ax=ax)
    ax.set_xlabel("Coefficient values corrected by the feature's std. dev.")
    ax.set_title("Ridge model, small regularization")
    ax.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.3)
    st.pyplot(fig)

# Trigger to move from Step 9 to Step 10
if st.session_state.step == 10:
    if st.button("YES - Proceed to Coefficient Variability", key="step10"):
        st.session_state.step = 11
    else:
        st.info("Click YES to proceed to 11. Coefficient Variability via Cross-Validation.")
        st.stop()


# ---------------------------
# STEP 11: Coefficient Variability via Cross-Validation
# ---------------------------
if st.session_state.step >= 11:
    st.header("11. Coefficient Variability via Cross-Validation")

    from sklearn.model_selection import RepeatedKFold, cross_validate

    # Define repeated k-fold CV
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)

    # Run CV and keep all fitted estimators
    cv_model = cross_validate(
        model,
        X,
        y,
        cv=cv,
        return_estimator=True,
        n_jobs=2,
    )

    # Collect coefficients corrected by feature std. dev. for each fold
    coefs_cv = pd.DataFrame(
        [
            est[-1].regressor_.coef_ * est[:-1].transform(X.iloc[train_idx]).std(axis=0)
            for est, (train_idx, _) in zip(cv_model["estimator"], cv.split(X, y))
        ],
        columns=feature_names,
    )

    # Plot variability
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.stripplot(data=coefs_cv, orient="h", palette="dark:k", alpha=0.5, ax=ax)
    sns.boxplot(data=coefs_cv, orient="h", color="cyan", saturation=0.5, whis=10, ax=ax)
    ax.axvline(x=0, color=".5")
    ax.set_xlabel("Coefficient importance")
    ax.set_title("Coefficient importance and its variability")
    plt.suptitle("Ridge model, small regularization")
    plt.subplots_adjust(left=0.3)
    st.pyplot(fig)

# Trigger to move from Step 11 to Step 12
if st.session_state.step == 11:
    if st.button("YES - Proceed to Co-variation of AGE and EXPERIENCE", key="step11"):
        st.session_state.step = 12
    else:
        st.info("Click YES to proceed to 12. Co-variation of AGE and EXPERIENCE.")
        st.stop()


# ---------------------------
# STEP 12: Co-variation of AGE and EXPERIENCE Coefficients
# ---------------------------
if st.session_state.step >= 12:
    st.header("12. Co-variation of AGE and EXPERIENCE Coefficients Across Folds")

    # Scatter plot of AGE vs EXPERIENCE coefficients from CV results
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coefs_cv["AGE"], coefs_cv["EXPERIENCE"], alpha=0.7)
    ax.set_ylabel("Age coefficient")
    ax.set_xlabel("Experience coefficient")
    ax.grid(True)
    ax.set_xlim(-0.4, 0.5)
    ax.set_ylim(-0.4, 0.5)
    ax.set_title("Co-variations of coefficients for AGE and EXPERIENCE across folds")
    st.pyplot(fig)

# Trigger to move from Step 12 to Step 13
if st.session_state.step == 12:
    if st.button("YES - Proceed to Impact of Dropping AGE on Model Stability", key="step12"):
        st.session_state.step = 13
    else:
        st.info("Click YES to proceed to 13. Impact of Dropping AGE on Model Stability")
        st.stop()


# ---------------------------
# STEP 13: Drop AGE to Test Collinearity Impact
# ---------------------------
if st.session_state.step >= 13:
    st.header("13. Impact of Dropping AGE on Model Stability")

    # Drop AGE and re-run CV
    column_to_drop = ["AGE"]
    cv_model_age_dropped = cross_validate(
        model,
        X.drop(columns=column_to_drop),
        y,
        cv=cv,
        return_estimator=True,
        n_jobs=2,
    )

    # Compute coefficients corrected by feature std. dev.
    coefs_age_dropped = pd.DataFrame(
        [
            est[-1].regressor_.coef_
            * est[:-1].transform(X.drop(columns=column_to_drop).iloc[train_idx]).std(axis=0)
            for est, (train_idx, _) in zip(cv_model_age_dropped["estimator"], cv.split(X, y))
        ],
        columns=feature_names[:-1],  # AGE removed
    )

    # Plot variability
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.stripplot(data=coefs_age_dropped, orient="h", palette="dark:k", alpha=0.5, ax=ax)
    sns.boxplot(data=coefs_age_dropped, orient="h", color="cyan", saturation=0.5, ax=ax)
    ax.axvline(x=0, color=".5")
    ax.set_title("Coefficient importance and its variability")
    ax.set_xlabel("Coefficient importance")
    plt.suptitle("Ridge model, small regularization, AGE dropped")
    plt.subplots_adjust(left=0.3)
    st.pyplot(fig)

# Trigger to move from Step 13 to Step 14
if st.session_state.step == 13:
    if st.button("YES - Proceed to Final Model Performance Check", key="step13"):
        st.session_state.step = 14
    else:
        st.info("Click YES to proceed to 14. Final Model Performance Check")
        st.stop()


# ---------------------------
# STEP 14: Final Model Performance Check
# ---------------------------
if st.session_state.step >= 14:
    st.header("14. Final Model Performance Check")

    # Calculate metrics
    mae_train = median_absolute_error(y_train, model.predict(X_train))
    y_pred = model.predict(X_test)
    mae_test = median_absolute_error(y_test, y_pred)
    scores = {
        "MedAE on training set": f"{mae_train:.2f} $/hour",
        "MedAE on testing set": f"{mae_test:.2f} $/hour",
    }
    st.write(scores)

    # Prediction error plot
    fig, ax = plt.subplots(figsize=(5, 5))
    PredictionErrorDisplay.from_predictions(
        y_test, y_pred, kind="actual_vs_predicted", ax=ax, scatter_kwargs={"alpha": 0.5}
    )
    ax.set_title("Ridge model, small regularization")
    for name, score in scores.items():
        ax.plot([], [], " ", label=f"{name}: {score}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)

# Trigger to move from Step 14 to Step 14
if st.session_state.step == 14:
    if st.button("YES - Proceed to Final Closure", key="step14"):
        st.session_state.step = 15
    else:
        st.info("Click YES to FINAL CLOSURE OF APP.")
        st.stop()



st.info("You have reached the end of the app. Thank you!")