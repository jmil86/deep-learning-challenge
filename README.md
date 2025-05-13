# deep-learning-challenge

## Alphabet Soup Deep Learning Model Analysis

### Overview of Analysis

The purpose of this analysis was to build, train, and evaluate a deep learning model capable of predicting whether applicants funded by Alphabet Soup will be successful or not. The model uses historical data from charitable organizations and applies a binary classification approach using a neural network to make this determination.

---

## Results

### Data Preprocessing

* **Target Variable**:

  * `IS_SUCCESSFUL` – This binary column indicates whether an applicant was successful (`1`) or not (`0`).

* **Feature Variables**:

  * All other columns, once categorical features were converted using `pd.get_dummies()`, including:

    * `APPLICATION_TYPE`
    * `AFFILIATION`
    * `CLASSIFICATION`
    * `USE_CASE`
    * `ORGANIZATION`
    * `STATUS`
    * `INCOME_AMT`
    * `SPECIAL_CONSIDERATIONS`
    * `ASK_AMT`

* **Dropped Columns**:

  * `EIN` and `NAME` – These columns were removed as they are identifiers and do not contribute to the model's learning process.

---

### Compiling, Training, and Evaluating the Model

* **Model Architecture**:

  * **Input Layer**: Automatically set based on the number of features (43 after preprocessing).
  * **First Hidden Layer**: 80 neurons, ReLU activation
  * **Second Hidden Layer**: 30 neurons, ReLU activation
  * **Output Layer**: 1 neuron, Sigmoid activation (for binary classification)

* **Model Compilation**:

  * Loss Function: `binary_crossentropy`
  * Optimizer: `adam`
  * Metrics: `accuracy`

* **Model Training**:

  * Trained over **100 epochs**
  * Achieved final **accuracy of \~72.6%** on the test set

* **Steps Taken to Improve Performance**:

  * Dropped low-variance categorical values by consolidating rare entries into `"Other"` (e.g., in `APPLICATION_TYPE`, `CLASSIFICATION`)
  * Applied one-hot encoding for categorical variables
  * Used feature scaling (`StandardScaler`)
  * Selected appropriate neuron counts and activations for balanced complexity
  * Added more neurons and epochs for added analysis and comparison

---

### Summary and Recommendation

* **Summary**:

  * The final model achieved an accuracy of approximately **72.6%**, which is **just short of the 75% target**.
  * The model appears to generalize reasonably well given the size and variability in the dataset, but additional improvements are possible.

* **Recommendation**:
  To improve performance further or try alternative approaches, consider:

  * **Random Forest Classifier** or **Gradient Boosted Trees**:

    * These are often more effective on tabular data with many categorical variables.
    * They handle non-linear relationships and feature importance naturally.
  * **Hyperparameter Tuning**:

    * Use `KerasTuner` or `GridSearchCV` (for non-DL models) to optimize neuron counts, learning rate, and layer architecture.
  * **Feature Engineering**:

    * Introduce domain-specific features or transformations, especially around `ASK_AMT` and `INCOME_AMT`.

