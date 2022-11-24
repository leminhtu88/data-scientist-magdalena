# Import of necessary packages
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
from random import randint
import seaborn as sns

# %% Variables & Switches 
do_preprocessing = 1

# %% Functions
# This function is used to calculate a result table including the relative and absolute deviation between two values
def create_resultstable(y_pred, y_test):
    results = pd.DataFrame([y_pred, y_test], index = ['y_predicted','y_test']).T
    results['deviations'] = results['y_predicted'] - results['y_test']
    results['rel_deviation'] = results['deviations']/results['y_test']
    results['abs_rel_deviation'] = abs(results['deviations']/results['y_test'])
    return results

# This function analyzes the deviation result table in cases of quantity splits
def table_deviation_per_quantity(results_table):
    dev_1 = results_table[results_table['y_test']<50]['abs_rel_deviation']
    dev_2 = results_table[(results_table['y_test']>50) & (results_table['y_test']<100)]['abs_rel_deviation']
    dev_3 = results_table[(results_table['y_test']>100) & (results_table['y_test']<150)]['abs_rel_deviation']
    dev_4 = results_table[(results_table['y_test']>150) & (results_table['y_test']<200)]['abs_rel_deviation']
    dev_5 = results_table[(results_table['y_test']>200)]['abs_rel_deviation']
    dev_quantity = pd.DataFrame([dev_1.describe(), dev_2.describe(), dev_3.describe(), dev_4.describe(), dev_5.describe()],
            index = ['0-50','50-100','100-150','150-200','>200']).T
    return dev_quantity

# This function plots the relative deviations
def plot_relative_deviation(results, ylimit):
    plt.figure(figsize = (12,8))
    plt.scatter(x = results['y_test'].values, y = results['rel_deviation'].values*100, s = 8)
    plt.xlabel('Q [mm^3]')
    plt.ylabel('Rel. error [%]')
    plt.plot([0,250], [3.6, 3.6], 'k-')
    plt.plot([0,250], [-3.6, -3.6], 'k-')
    axes = plt.gca()
    axes.set_ylim([-ylimit,ylimit])
    plt.show()
    
# Model training incl. plotting the result deviation
def percentage_split_eval(regression_model, X_train, y_train, ylimit):
    regression_model.fit(X_train, y_train)
    y_predicted = regression_model.predict(X_train)
    results = create_resultstable(y_predicted, y_train)
    print("Mean absolute error on training set", 
          mean_absolute_error(regression_model.predict(X_train), y_train))
    print("")
    results_per_quantity = table_deviation_per_quantity(results)
    display(results_per_quantity)
    plot_relative_deviation(results, ylimit)
    return(results_per_quantity)

# Tsting of model incl. plotting the result deviation
def percentage_split_test(regression_model, X_test, y_test, ylimit):
    y_predicted = regression_model.predict(X_test)
    results = create_resultstable(y_predicted, y_test)
    print("Mean absolute error on test set",mean_absolute_error(y_pred=y_predicted, y_true=y_test))
    results_per_quantity = table_deviation_per_quantity(results)
    display(results_per_quantity)
    plot_relative_deviation(results, ylimit)
    return(results_per_quantity)
    
# %% Load Data
#Training Data
dataTrain = pd.read_csv('../data/Trainingdata_p4m.csv',';')
# Test Data
dataTest = pd.read_csv('../data/Testdata_p4m.csv',';')

# %% Data Preprocessing
if do_preprocessing==1:
    # Correct by maximum pressure
    for k in dataTrain.index:
        dataTrain.loc[k,"p_transient_1 [bar]":"p_transient_20 [bar]"]=dataTrain.loc[k,"p_transient_1 [bar]":"p_transient_20 [bar]"].values-dataTrain.loc[k,"p_transient_1 [bar]":"p_transient_20 [bar]"].values.max()
    for l in dataTest.index:
        dataTest.loc[l,"p_transient_1 [bar]":"p_transient_20 [bar]"]=dataTest.loc[l,"p_transient_1 [bar]":"p_transient_20 [bar]"].values-dataTest.loc[l,"p_transient_1 [bar]":"p_transient_20 [bar]"].values.max()

# %% Data analysis
# Analyse names of Data Frame columns
print(dataTrain.columns)
# Analyse values of each column
print(dataTrain.describe())
# Plot for a given number pressure curves
number_of_plots = 2
random_numbers = [randint(0,dataTrain.shape[0]) for p in range(0,number_of_plots)]
print(random_numbers)
for i in random_numbers:
    plt.plot(dataTrain.loc[i,"p_transient_1 [bar]":"p_transient_20 [bar]"].values)
    plt.show()
    print("Q:", dataTrain.loc[i,"Q_Inj [mm3]"])
    print("Max:", dataTrain.loc[i,"p_transient_1 [bar]":"p_transient_20 [bar]"].values.max(), "Min:", dataTrain.loc[i,"p_transient_1 [bar]":"p_transient_20 [bar]"].values.min(),
         "Diff:", dataTrain.loc[i,"p_transient_1 [bar]":"p_transient_20 [bar]"].values.max() - dataTrain.loc[i,"p_transient_1 [bar]":"p_transient_20 [bar]"].values.min())
    print("Max_ber:", dataTrain.loc[i,"p_transient_1 [bar]":"p_transient_20 [bar]"].values.max(), "Min_ber:", dataTrain.loc[i,"p_transient_1 [bar]":"p_transient_20 [bar]"].values.min())
# Calculate correlation coefficients for two columns
Coeff = np.corrcoef(dataTrain['Q_Inj [mm3]'], dataTrain['p_transient_5 [bar]'])
print(Coeff)
#Plotting a correlation plot (all in one)
plt.rcParams["figure.figsize"] = (10,10)
sns.heatmap(dataTrain.corr(), annot=True, cmap="RdYlGn")
plt.show()
# Plot correlation of two columns
plt.plot(dataTrain['Q_Inj [mm3]'], dataTrain['p_transient_5 [bar]'], 'bo')
# Create a scatter matrix from the dataframe (all in one)
grr = scatter_matrix(dataTrain, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)

# %% Regression model building
# Define input & output variables used for training
xTrain = dataTrain.loc[:,"nEng [rpm]":"p_transient_20 [bar]"].values
yTrain = dataTrain['Q_Inj [mm3]']
# Scale input variables to range from -1 to 1
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(xTrain)
xTrain_scaled = pd.DataFrame(scaler.transform(xTrain))
# Set model parameters for Neural Network
alpha = 0.0001
neurons = (3,)
solver_chosen = 'lbfgs'
activation_function = 'logistic'
maximum_iterations = 10000
simple_mlp_reg = MLPRegressor(hidden_layer_sizes=neurons, activation=activation_function, solver = solver_chosen, max_iter=maximum_iterations)
# Calculating & evaluating regression with training data
table_per_quantity = percentage_split_eval(simple_mlp_reg, xTrain_scaled, yTrain, 20)

# %% Testing regression model
# Define input & output variables used for testing (must be equal to training variables)
xTest = dataTest.loc[:,"nEng [rpm]":"p_transient_20 [bar]"].values
yTest = dataTest['Q_Inj [mm3]']
# Scale input variables to range from -1 to 1
xTest_scaled = pd.DataFrame(scaler.transform(xTest))
# Evaluating regression with test data
table_per_quantity = percentage_split_test(simple_mlp_reg, xTest_scaled, yTest, 200)

# %% Outlier detection
## Set parameters for nearest neighbors
number_of_neighbors=10
# Fit model on training data
nbrs = NearestNeighbors(n_neighbors=number_of_neighbors, algorithm='ball_tree').fit(xTrain_scaled)
# Calculate distances of test data compared to model
distances, neighbourpoints = nbrs.kneighbors(xTest_scaled)
# Define distance to split data in outlier and non-outlier
allowed_distance = 0.3
# Split test data in outlier and non-outlier
xTest_scaled_out = xTest_scaled[distances[:,9]>allowed_distance]
yTest_out = yTest[distances[:,9]>allowed_distance]
xTest_scaled_no_out = xTest_scaled[distances[:,9]<allowed_distance]
yTest_no_out = yTest[distances[:,9]<allowed_distance]
# Test with outlier and non-outlier
# Outlier
table_per_quantity = percentage_split_test(simple_mlp_reg, xTest_scaled_out, yTest_out, 200)
# Non-Outlier
table_per_quantity = percentage_split_test(simple_mlp_reg, xTest_scaled_no_out, yTest_no_out, 200)


    