# -*- coding: utf-8 -*-
"""
NIH NCATS Bias Detection Challenge 
Bias Detection Tool for Researchers
Team: ZIRI
"""


from tkinter import *
import tkinter.filedialog as filedialog
import pandas as pd
from scipy.spatial.distance import jensenshannon
import numpy as np
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from tqdm import tqdm
import pickle
from tkinter import ttk
import sys


#Class that creates the object representing our Bias Detection Tool
class Tool(Frame):
    protected_groups = []
    reference_groups = []
    all_groups = []
    labeling_variables = []
    indicated_outcomes = ""
    data = pd.DataFrame()
    member_cols = []
    py_model = ""

    
    
    #Function to initiate the Tkinter window and call instantiateGUI
    def __init__(self):
        self.root = Tk()
        self.instantiateGUI()


    
    #Helper function to pop up file explorer for user to select CSV (May be possible to perform logic in instantiateGUI - revisit)
    def fileExplorer(self, csv_file_label):
        file_path = filedialog.askopenfilename(initialdir = "/", title = "Select file", filetypes = (("CSV files", "*.csv"), ("all files", "*.*")))
        csv_file_label.config(text=file_path)
        self.data = pd.read_csv(file_path)


    #Function to save the user inputs into class variables and update the labels on the GUI            
    def getUserInputs(self, protected_groupings_input, reference_groupings_input, protected_variables_label, reference_variables_label, labeling_attribute_input, labeling_attribute_label, indicated_outcomes_input, indicated_outcomes_label):
        user_input = protected_groupings_input.get(1.0, "end-1c")
        self.protected_groups.append(user_input)
        output_string = ""
        for group in self.protected_groups:
            output_string += group + ", "
        protected_variables_label.config(text = "Protected Groups: " + output_string[:-2])
        user_input_reference = reference_groupings_input.get(1.0, "end-1c")
        self.reference_groups.append(user_input_reference)
        output_string2 = ""
        output_string3 = ""
        for group2 in self.reference_groups:
            output_string2 += group2 + ", "
        reference_variables_label.config(text = "Reference Groups: " + output_string2[:-2])   
        user_input_labeling = labeling_attribute_input.get(1.0, "end-1c")
        self.labeling_variables.append(user_input_labeling)
        for group3 in self.labeling_variables:
            output_string3 += group3 + ", "
        labeling_attribute_label.config(text = "Labeling Variables: " + output_string3[:-2])
        user_input_outcomes = indicated_outcomes_input.get(1.0, "end-1c")
        self.indicated_outcomes = user_input_outcomes
        indicated_outcomes_label.config(text = "Favored Outcome: " + self.indicated_outcomes)
        
        
    #Main function to create and display the GUI to the user    
    def instantiateGUI(self):
        self.root.title("Bias Detection tool for Research")
        #Create the left frame containing buttons for user file selection and protected group input
        left_frame = Frame(self.root)
        left_frame.grid(row = 1, column = 1)
        csv_file_label = Label(left_frame, text="No file selected")
        csv_file_button = Button(left_frame, text="Select CSV File", command= lambda : self.fileExplorer(csv_file_label))
        csv_file_button.pack()
        csv_file_label.pack(fill = BOTH)
        protected_variables_label = Label(left_frame, text="Protected Variables:")
        protected_variables_label.pack()
        protected_groupings_input = Text(left_frame, height = 5, width = 20)
        protected_groupings_input.pack(fill = BOTH)
        reference_variables_label = Label(left_frame, text = "Reference Variables:")
        reference_variables_label.pack()
        reference_groupings_input = Text(left_frame, height = 5, width = 20)
        reference_groupings_input.pack(fill=BOTH)
        labeling_attribute_label = Label(left_frame, text = "Labeling Variable:")
        labeling_attribute_label.pack()
        labeling_attribute_input = Text(left_frame, height = 5, width = 20)
        labeling_attribute_input.pack(fill = BOTH)
        indicated_outcomes_label = Label(left_frame, text = "Indicate favored outcome:")
        indicated_outcomes_label.pack()
        indicated_outcomes_input = Text(left_frame, height = 5, width = 20)
        indicated_outcomes_input.pack(fill = BOTH)
        save_groups_button = Button(left_frame, text="Save Groups", command = lambda : self.getUserInputs(protected_groupings_input, reference_groupings_input, protected_variables_label, reference_variables_label, labeling_attribute_input, labeling_attribute_label, indicated_outcomes_input, indicated_outcomes_label))
        save_groups_button.pack()

              
        #Create the right frame containing the results, wait to instantiate it when we have data
        right_frame = Frame(self.root)
        right_frame.grid(row=1, column = 2)
        data_metrics_label = Label(right_frame, text = "Data level Bias Metrics")
        data_metrics_output1 = ttk.Treeview(right_frame)
        data_metrics_output2 = ttk.Treeview(right_frame)
        data_metrics_output3 = ttk.Treeview(right_frame)
        algorithm_metrics_label = Label(right_frame, text = "Algorithm level Bias Metrics")
        algorithm_metrics_output = ttk.Treeview(right_frame)
        run_button = Button(left_frame, text = "Run", command = lambda : self.getBias(algorithm_metrics_output, data_metrics_output1, data_metrics_output2, data_metrics_output3,
                                                                                      data_metrics_label, algorithm_metrics_label))
        run_button.pack()
      
    #Function to compute the demographic disparity in the dataset  
    #Returns the conditional demographic disparity for all groups and an array of outcomes indicating the demographic disparities between each group
    def findDemographicDisparity(self):
        outcomes = []
        demographic_disparities = []
        #CDD = (1/n) * Summation(ini) * DDi
        #DDd = Nd(0) / N(0) - Nd(1) / N(1)
        #Lets start by finding the total # of rejected and accepted outcomes for each cohort
        for col in self.member_cols:
            for group in self.all_groups:
                group_df = self.data[self.data[col] == group]
                #Will need to find a better way of handling the indicated outcomes, currently assuming the outcome is 
                #binary (1 or 0)
                Num_accepted = group_df[group_df[self.labeling_variables[0]] == int(self.indicated_outcomes)]
                Num_rejected = group_df[group_df[self.labeling_variables[0]] != int(self.indicated_outcomes)]
                outcomes.append([group, len(Num_accepted), len(Num_rejected)])
        #Now that we have accepted and rejected outcomes for each group we compute the Demographic Disparity
        for outcome in outcomes:
            group_label = outcome[0]
            group_accepted = outcome[1]
            group_rejected = outcome[2]
            group_total = group_accepted + group_rejected
            Pdr = group_rejected / group_total
            Pda = group_accepted / group_total
            Dd = Pdr - Pda
            demographic_disparities.append([group_label, group_total, Dd])
        #Now compute the CDD
        total_observations = 0
        running_CDD = 0
        for demographic in demographic_disparities: #Summation of Ni * DDi
            total_observations += demographic[1]
            running_CDD += (demographic[1] * demographic[2])
        CDD = (1/total_observations) * running_CDD
        return CDD, demographic_disparities
        

    #Function to compute the Jensen Shannon Divergence, returns an array indicating the jsd between each group
    def findJensenShannonDivergence(self):
        group_divergence = []
        grouped = self.data.groupby(self.member_cols) #Group columns together where indicated groupings are found      
        probabilities = grouped[self.labeling_variables[0]].value_counts(normalize = True) #Get counts for positive and negative outcome, normalize them within 0 to 1
        probabilities = probabilities.unstack() #Clean DF up
        probabilities = probabilities.fillna(0) #Handle missing values
        # Calculate the Jensen-Shannon divergence between each pair of distributions
        for i in range(len(probabilities)):
            for j in range(i+1, len(probabilities)):
                p = probabilities.iloc[i]
                q = probabilities.iloc[j]
                jsd = jensenshannon(p, q)
                group_divergence.append([p.name, q.name, jsd])
        return group_divergence
        
        
    #Function to compute the class imbalance between each group, returns an array containing the class imbalance for each group combination    
    def findClassImbalance(self):
        #CI = (Na - Nd) / (Na + Nd) where Na and Nd are different cohorts
        group_counts = []
        class_counts = []
        #Start with Protected Group
        for member in self.protected_groups:
            #Find columns where group is found
            self.member_cols = self.searchCSV(member)            
            for col in self.member_cols: #Count the number of instances of each value in the each of the columns
                count = self.data[col].value_counts()
                group_counts.append(count)  
        #Group_counts contains a series object, loop through each series
        for i in range(0, len(group_counts)):
            series = group_counts[i].values.astype(int) 
            series_label = group_counts[i].index.tolist()
            for j in range(0, len(series)): #For each of the groups, find Na 
                Na = series[j]
                Na_label = series_label[j]
                self.all_groups.append(Na_label)
                for t in range(j+1, len(series)): #For each of the groups except Na group, find Nd and calculate CI
                    Nd = series[t]
                    Nd_label = series_label[t]
                    class_imbalance = (Na - Nd) / (Na + Nd)
                    class_counts.append([Na_label, Nd_label, class_imbalance])
        class_counts_df = pd.DataFrame(class_counts, columns = ['Group 1', "Group 2", "Class Imbalance"])
        self.all_groups = list(set(self.all_groups))
        return class_counts_df
            
    
    #Helper function for class imbalance, searches for every instance of the search_string and returns the columns 
    #where each of instance of the word is found
    def searchCSV(self, search_string):
        matching_columns = []
        for index, row in self.data.iterrows():
            for col in self.data.columns:
                if search_string in str(row[col]):
                    matching_columns.append(col)       
        #Remove duplicates
        matching_columns = list(set(matching_columns))        
        return matching_columns


    #Function to preprocess some of the data and call each of the relevant metrics functions, displays each metric inside a Tkinter treeview
    def getBias(self, algorithm_metrics_output, data_metrics_output1, data_metrics_output2, data_metrics_output3, data_metrics_label, algorithm_metrics_label):
        #Start with Data Level Biases
        CI = self.findClassImbalance() #Get Class Imbalance metrics
        data_metrics_output1['columns'] = list(CI.columns) #Set columns of output treeview to columns of CI dataframe
        for col1 in CI.columns:
            data_metrics_output1.heading(col1, text = col1)
        for l, row in CI.iterrows(): #populate treeview with values from CI
            data_metrics_output1.insert('', index = 0, text = l, values = list(row))
        data_metrics_label.pack()
        data_metrics_output1.pack(fill = "both", expand = True)
                
        
        #Get Demographic Disparity Metrics
        CDD, demographic_disparities = self.findDemographicDisparity()
        demographic_disparities.append(['All Groups CDD', 0, CDD])
        #Convert to dataframe
        demo_disparity_df = pd.DataFrame(demographic_disparities, columns=['Group', 'total count', 'Demographic Disparity'])
        demo_disparity_df = demo_disparity_df.drop('total count', axis = 1) #drop middle column containing group counts
        data_metrics_output2['columns'] = list(demo_disparity_df.columns)
        for col2 in demo_disparity_df.columns:
            data_metrics_output2.heading(col2, text = col2)
        for i, row2 in demo_disparity_df.iterrows():
            data_metrics_output2.insert('', index = 0, text = i, values = list(row2))
        data_metrics_output2.pack(fill = "both", expand = True)
        
        #Get Jensen Shannon Divergence
        group_divergence = self.findJensenShannonDivergence()
        #convert to DF
        divergence_df = pd.DataFrame(group_divergence, columns = ['Group 1', 'Group 2', 'JSD'])
        data_metrics_output3['columns'] = list(divergence_df.columns)
        for col3 in divergence_df:
            data_metrics_output3.heading(col3, text = col3)
        for m, row3 in divergence_df.iterrows():
            data_metrics_output3.insert('', index = 0, text = m, values = list(row3))
        data_metrics_output3.pack(fill = "both", expand = True)
        
        #Get biases at Algorithm level
        model_preds, cat_data = self.algorithmicBias()
        dataset = StandardDataset(cat_data, label_name = self.labeling_variables[0],
                                  favorable_classes = [1], protected_attribute_names = self.member_cols, privileged_classes = [[0]])
        results = self.model_fairness(dataset, model_preds)
        #Start of code to print results to the UI
        df_results = results.data
        algorithm_metrics_output['columns'] = list(df_results.columns)
        for col in df_results.columns:
            algorithm_metrics_output.heading(col, text = col)
        for i, row in df_results.iterrows():
            algorithm_metrics_output.insert('', 'end', text = i, values = list(row))
        algorithm_metrics_label.pack()
        algorithm_metrics_output.pack()
     
        
        
        
    #Calculates and returns the prediction values for the ML models
    def algorithmicBias(self):
        #Select and categorize columns in data
        cat_columns = self.data.select_dtypes(['object']).columns
        cat_data = self.data.copy()
        cat_data[cat_columns] = self.data[cat_columns].apply(lambda x: pd.factorize(x)[0])
        #Find mapping for bias detection
        map_dict = {}
        for item in cat_columns:
            map_dict[item] = set(list(zip(self.data[item], cat_data[item])))\
        #Reformat data for classification
        columns_to_group = self.data.columns.difference([self.labeling_variables[0]]).tolist()
        X = cat_data[columns_to_group].to_numpy()
        Y = cat_data[self.data.columns.difference(columns_to_group)].to_numpy()
        # Train and testing split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)
        #Lazy predict - train multiple ML models
        clf = LazyClassifier(verbose = 0, ignore_warnings = True, custom_metric = matthews_corrcoef)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        #Get the model names
        model_dict = clf.provide_models(X_train, X_test, y_train, y_test)
        #Generic estimator extractor for scikit learn
        CLASSIFIERS = [est for est in all_estimators() if (issubclass(est[1], ClassifierMixin))]
        #Get predictions from top n models
        rank_range = 15
        model_preds = {}
        for i in tqdm(range(1, rank_range+1)):
            try:
                clf_name = self.return_best_clf(i, CLASSIFIERS, models)[0]
                clf_func = self.return_best_clf(i, CLASSIFIERS, models)[1]()
                #predict on all data
                y_pred = clf_func.fit(X, Y).predict(X)
                model_preds[clf_name] = y_pred
            except:
                pass
        return model_preds, cat_data
            
   #Function to further format the data and call the aif360 package to compute the metrics         
    def fair_metrics(self, dataset, y_pred):
        dataset_pred = dataset.copy()
        dataset_pred.labels = y_pred
        attr = dataset_pred.protected_attribute_names[0]
        idx = dataset_pred.protected_attribute_names.index(attr)
        privileged_groups = [{attr:dataset_pred.privileged_protected_attributes[idx][0]}]
        unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}]
        classified_metric = ClassificationMetric(dataset, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        result = {'Statistical_parity_difference': "{:e}".format(metric_pred.statistical_parity_difference()),
                  'Disparate_impact': "{:e}".format(metric_pred.disparate_impact()),
                  'Equal_opportunity_difference': "{:e}".format(classified_metric.equal_opportunity_difference()),
                  'Average_odds_difference':"{:e}".format(classified_metric.average_odds_difference()),
                  'Theil Index':"{:e}".format(classified_metric.generalized_entropy_index(alpha=1)) #theil index is generalized entropy index with alpha=1
                  }
        return result        
        
        
    #Fairness metrics for all the top models    
    def model_fairness(self, dataset, model_preds):
        #Get the analysis
        attr = dataset.protected_attribute_names[0]
        model_name = []
        l_statistical_parity_difference = []
        l_disparate_impact = []
        l_equal_opportunity_difference = []
        l_average_odds_difference = []
        l_theil_index = []
        
        for mod in model_preds:
            model_name.append(mod)
            #print(mod+' :',fair_metrics(dataset,model_preds[mod]))
            vals = self.fair_metrics(dataset,model_preds[mod])
            l_statistical_parity_difference.append(vals['Statistical_parity_difference'])
            l_disparate_impact.append(vals['Disparate_impact'])
            l_equal_opportunity_difference.append(vals['Equal_opportunity_difference'])
            l_average_odds_difference.append(vals['Average_odds_difference'])
            l_theil_index.append(vals['Theil Index'])
            scores = pd.DataFrame({
                        "Model": model_name,
                        "Statistical Parity Difference": l_statistical_parity_difference,
                        "Disparate Impact": l_disparate_impact,
                        "Equal Opportunity Difference": l_equal_opportunity_difference,
                        "Average Odds Difference": l_average_odds_difference,
                        "Theil Index": l_theil_index,
                    })
            scores = scores.set_index("Model")
            scores = scores.style.set_caption('Model Bias Detection for variable = ' + attr)  
        return scores
        
        
        
        
        
    #Helper function to return the top classifiers for the lazypredict package    
    def return_best_clf(self, rank, CLASSIFIERS, models):
        for clf_item in CLASSIFIERS:
            if clf_item[0] == models.iloc[rank-1].name:
                return (clf_item)
        
    


def main():
    app = Tool()
    app.root.mainloop()
    

main()