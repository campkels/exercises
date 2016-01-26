#---------------------------------------------------------------------------------------------------------
# RTI CDS Analytics Exercise 01 - Kelsey Campbell - 1/25/2016
#---------------------------------------------------------------------------------------------------------

#------------------------------------------------------------
# Import Packages
#------------------------------------------------------------
import sqlite3
import pandas as pd
from sklearn import cross_validation, linear_model, metrics
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

#------------------------------------------------------------------------------
# 
#   CREATE DATA
#		
# -----------------------------------------------------------------------------

#------------------------------------------------------------
# Connect to Database
#------------------------------------------------------------
conn = sqlite3.connect('C:\Users\Kelsey\Google Drive\jobs\RTI\exercises\exercise01\exercise01.sqlite')
c = conn.cursor()

#------------------------------------------------------------
#  Create Flat Table
#------------------------------------------------------------

# Query Main Records Table, Create List of Fields
#-------------------------------------------------
rec = pd.read_sql_query('SELECT * FROM records', conn)
recfields = ",".join(list(rec.columns.values)[1:])

# Join Main Table with Value Tables
#-------------------------------------------------
flat = pd.read_sql_query('SELECT records.id as id, %s, races.name as race_str, countries.name as country_str, workclasses.name as workclass_str, \
 								  education_levels.name as education_level_str, marital_statuses.name as marital_status_str, \
 								  occupations.name as occupation_str, sexes.name as sex_str, relationships.name as relationship_str \
 						  FROM records LEFT JOIN races ON records.race_id = races.id \
						  			   LEFT JOIN countries ON records.country_id = countries.id \
						  			   LEFT JOIN workclasses ON records.workclass_id = workclasses.id \
						  			   LEFT JOIN education_levels ON records.education_level_id = education_levels.id \
						  			   LEFT JOIN marital_statuses ON records.marital_status_id = marital_statuses.id \
						  			   LEFT JOIN occupations ON records.occupation_id = occupations.id \
						  			   LEFT JOIN sexes ON records.sex_id = sexes.id \
						  			   LEFT JOIN relationships ON records.relationship_id = relationships.id' %(recfields), conn)

# Save Dataframe to CSV
#-------------------------------------------------
flat.to_csv('C:\Users\Kelsey\Google Drive\jobs\RTI\exercises\exercise01\\flat.csv')
conn.close()

#------------------------------------------------------------
#  Explore Variables
#------------------------------------------------------------

# 1 Way Frequency Counts of Categorical Variables
#-------------------------------------------------
for column in flat:
	if flat[column].dtypes == 'O':
		print
		print('Frequency Table for ' + column)
		print(flat[column].value_counts())

# Summary Stats, Histograms for Continous Variables
#-------------------------------------------------
for column in flat:
	if column[-2:] != 'id' and flat[column].dtypes == 'int64':
		print
		print('Summary Stats for ' + column)
		print(flat[column].describe())
		
#train.hist()
#plt.show()

#------------------------------------------------------------
#  Prepare Data for Analysis 
#------------------------------------------------------------

# Functions to Bin/Recatagorize Some Variables
#-------------------------------------------------
def native(data):
	if data['country_str'] == 'United-States':
		return 1
	else:
		return 0

def marital_stat(data):
	if data['marital_status_str'] in ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']:
		return "Married"
	else:
		return data['marital_status_str']

def agebin(data):
	if data['age'] < 20:
		return 'less20'
	elif 20 <= data['age'] < 30:
		return '20to29'
	elif 30 <= data['age'] < 40:
		return '30to39'
	elif 40 <= data['age'] < 50:
		return '40to49'
	elif 50 <= data['age'] < 60:
		return '50to59'
	elif 60 <= data['age'] < 70:
		return '60to69'
	elif data['age'] >= 70:
		return '70plus'

def educbin(data):
	if data['education_level_str'] in ['HS-grad', 'Some-college', 'Bachelors']:
		return data['education_level_str']
	elif data['education_level_str'] in ['Assoc-acdm', 'Assoc-voc']:
		return 'Associates'
	elif data['education_level_str'] in ['Doctorate', 'Masters', 'Prof-school']:
		return 'GradDeg'
	else:
		return 'LessthanHS'

def hrsbin(data):
	if data['hours_week'] < 10:
		return 'less10'
	elif 10 <= data['hours_week'] < 30:
		return '10to30'
	elif 30 <= data['hours_week'] < 50:
		return '30to50'
	elif data['hours_week'] >= 50:
		return '50plus'

def netcap(data):
	if data['capital_diff'] > 0:
		return 'NetGain'
	elif data['capital_diff'] == 0:
		return 'Zero'
	else:
		return 'NetLoss'

# Create Categorical Dummies
#-------------------------------------------------

# Replace Missings for Occupation Variable
flat['new_occupation_str'] = flat['occupation_str']
flat.loc[flat['occupation_str'] == '?', 'new_occupation_str'] = 'Missing'

# Take Difference of Capital Variables to get Net Gain/Loss
flat['capital_diff'] = flat['capital_gain'] - flat['capital_loss']

# Apply Rebinning Functions to Each Row in New Variables
flat['marital_stat'] = flat.apply(marital_stat, axis=1)
flat['agebin'] = flat.apply(agebin, axis=1)
flat['educbin'] = flat.apply(educbin, axis=1)
flat['hrsbin'] = flat.apply(hrsbin, axis=1)
flat['netcap'] = flat.apply(netcap, axis=1)

# Create Dummies
dumrace = pd.get_dummies(flat['race_str'],prefix='race') # Reference = White
dumsex = pd.get_dummies(flat['sex_str'],prefix='sex') # Reference = Male
dummarital_stat = pd.get_dummies(flat['marital_stat'],prefix='marital_stat') # Reference = Never Married
dumagebin = pd.get_dummies(flat['agebin'],prefix='agebin') # Reference = less20
dumeducbin = pd.get_dummies(flat['educbin'],prefix='educbin') # Reference = LessthanHS
dumoccu = pd.get_dummies(flat['new_occupation_str'],prefix='occu') # Reference = Adm-clerical
dumhrsbin = pd.get_dummies(flat['hrsbin'],prefix='hrsbin') # Reference = 30to50 (fulltime)
dumnetcap = pd.get_dummies(flat['netcap'],prefix='netcap') # Reference = 0

# Create Dataframe of Target Vars and X's of Interest
#-------------------------------------------------

# Combine Created Dummy Sets (Only Non-Reference Columns)
flat2 = flat[['over_50k', 'education_num']].join(dumrace.ix[:, :'race_Other']).join(dumsex[['sex_Female']]) \
		.join(dummarital_stat.ix[0:, [0,1,2,4]]).join(dumagebin.ix[:,:6]).join(dumeducbin.ix[0:, [0,1,2,3,5]]) \
		.join(dumoccu.ix[:,1:]).join(dumhrsbin.ix[0:, [0,2,3]]).join(dumnetcap.ix[:,:2])

# Apply Rebinning Functions for Binary Vars
flat2['native'] = flat.apply(native, axis=1)

#------------------------------------------------------------
#  Convert Pandas Dataframe to Numpy Array (for sklearn)
#------------------------------------------------------------

# Save Column Names for X's 
#-------------------------------------------------
xnames = list(flat2)[1:] 
xnames_man = list(flat2)[2:] 

# Convert
#-------------------------------------------------
xes = flat2[xnames].values
y = flat2[['over_50k']].values.ravel()

#------------------------------------------------------------
#  Split Data (Roughly 70%/20%/10% for Training, Validation, and Test)
#------------------------------------------------------------
x_train, x_temp, y_train, y_temp = cross_validation.train_test_split(xes, y, train_size=0.7, test_size=0.3)
x_valid, x_test, y_valid, y_test = cross_validation.train_test_split(x_temp, y_temp, train_size=0.67, test_size=0.33)

#------------------------------------------------------------------------------
# 
#   ANALYSIS
#		
# -----------------------------------------------------------------------------

#------------------------------------------------------------
#  Build Models 
#------------------------------------------------------------

# Subset Arrays to Reflect Desired Variables for each Model
#-------------------------------------------------

# Simple Model - Only Cont Education and Race Dummies
x_train1 = x_train[0:, 0:5]
x_valid1 = x_valid[0:, 0:5] 

# Manual Model - Exclude Cont Education, Keeps Everything Else from Above (All Categorical)
x_train2 = x_train[0:, 1:]
x_valid2 = x_valid[0:, 1:]

# Run Models
#-------------------------------------------------
for model in range(1,4):
	if model != 3:
		# Fit Specified Models
		exec "reg%s = linear_model.LogisticRegression()" % model
		exec "fit%s = reg%s.fit(x_train%s,y_train)" % (model, model, model)
	else:
		# Fit Reduced Model
		'''Not sure if Im doing this right. Wanted to play with the feature selection in sklearn, but it seems if just looks 
		   at the coefficient values to pick the most important, and once it cuts down your x's you lose what the final variables 
		   are? Maybe not great for this, but :) '''
		reg3 = linear_model.LogisticRegression()
		x_train3 = fit2.transform(x_train2)
		x_valid3 = fit2.transform(x_valid2)
		fit3 = reg3.fit(x_train3,y_train)
	# Score Models on Validation Data (cutoff is .5)
	exec "predicted%s = reg%s.predict(x_valid%s)" % (model, model, model)
	exec "probs%s = reg%s.predict_proba(x_valid%s)" % (model, model, model)
	# Create Comparison Points
	exec "M%s = [reg%s.score(x_train%s,y_train), metrics.accuracy_score(y_valid, predicted%s)]" % (model, model, model, model)
	exec "M%s_2 = [metrics.roc_auc_score(y_valid, probs%s[:, 1])]" % (model, model)

#------------------------------------------------------------
#  Compare Models
#------------------------------------------------------------

# Compare Prediction Accuracy
#-------------------------------------------------
datasets = ['Training', 'Validation']
print
print "Accuracy by Model and Dataset"
print pd.DataFrame({"Dataset":datasets, "Simple Model":M1, "Manual Model":M2, "Reduced Model":M3})

# Compare AUC
#-------------------------------------------------
print
print "Area Under ROC Curve by Model"
print pd.DataFrame({"Dataset":"Validation", "Simple Model":M1_2, "Manual Model":M2_2, "Reduced Model":M3_2})

#------------------------------------------------------------
#  Final Model Results on Test Data
#------------------------------------------------------------
''' Manual model scored the best on metrics '''

# Manual Model - Exclude Cont Education, Keeps Everything Else from Above (All Categorical)
#-------------------------------------------------
x_test2 = x_test[0:, 1:]

# Score Final Model on Test Data (cutoff is .5)
#-------------------------------------------------
predicted_fin = reg2.predict(x_test2)
probs_fin = reg2.predict_proba(x_test2)

# Final Model Metrics and Results
#-------------------------------------------------
print
print "Final Accuracy: " + str(metrics.accuracy_score(y_test, predicted_fin))
print "Final AUC: " + str(metrics.roc_auc_score(y_test, probs_fin[:, 1]))
print "Confusion Matrix:"
print metrics.confusion_matrix(y_test, predicted_fin)
print "Classification Report:"
print metrics.classification_report(y_test, predicted_fin)

#------------------------------------------------------------------------------
# 
#   REPORTING
#		
# -----------------------------------------------------------------------------

#------------------------------------------------------------
#  Final Model Results on Entire Dataset
#------------------------------------------------------------

# Save Final X's for Entire Dataset
#-------------------------------------------------
xes_fin = xes[0:, 1:]

# Run Final Model
#-------------------------------------------------
finmodel = linear_model.LogisticRegression()
finmodel.fit(xes_fin,y)

print "Final Coefficients and Odds Ratios"
print pd.DataFrame(zip(xnames_man, np.transpose(finmodel.coef_), np.exp(np.transpose(finmodel.coef_))))

# Run Final Model with Different Package
#-------------------------------------------------
''' I guess sklearn doesnt really do significance? Not cool '''

# Panda Dataframes of Target and Explanatory Vars
df_y = flat2['over_50k']
df_x = flat2.ix[:,2:]

# Manually Add the Intercept
df_x['intercept'] = 1.0

# Model
logit = sm.Logit(df_y, df_x)
result = logit.fit()

# Results
print result.summary() # 
print "Odds Ratios"
here = result.conf_int()
here['OR'] = result.params
here.columns = ['LB', 'UB', 'OR']
ORs = np.exp(here)
print ORs
'''Most of the estimates are close to sklearn results, age is different though. Not sure why. Data is fine.
OR's are way higher, finmodel.intercept_ is different though?
#bytreatment = flat.groupby('agebin')
#print bytreatment['over_50k'].describe()'''

#------------------------------------------------------------
#  Visualization
#------------------------------------------------------------

# Save Odds Ratios and CI's of "Changeagble" Attributes
#-------------------------------------------------
forplot = ORs.ix[15:34,:]

# Save Dataframe to CSV...Im going to R :)
#-------------------------------------------------
forplot.to_csv('C:\Users\Kelsey\Google Drive\jobs\RTI\exercises\exercise01\\forplot.csv')