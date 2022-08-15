import pandas as pd
import plotly.express as ex


def get_df():
    df = pd.read_csv('Data/marketing_campaign.csv', sep='\t', encoding='latin1')
    df.info()
    #plot the histogram of Year_Birth Distribution
    # Plot the Histogram of Year_Birth Distribution
    # figure = ex.histogram(df,
    #                       x='Year_Birth',
    #                       title='Year Birth Distribution',
    #                       labels={'x': 'Year Birth', 'y': 'Count'})
    #
    # figure.show()

    df['Age'] = 2022 - df['Year_Birth']
    # df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], infer_datetime_format=True)
    df['Income'] = df['Income'].fillna(df['Income'].mean())

    #Remove outlier on age column
    df = df.drop(df[df.Age>85].index)

    #Divide age of customer into 4 category
    cut_labels_Age = ['Young', 'Adult', 'Mature', 'Senior']
    cut_bins = [0, 30, 45, 65, 120]
    df['Age_group'] = pd.cut(df['Age'], bins=cut_bins, labels=cut_labels_Age)

    #divide education of customer into two categories
    df['Education']=df['Education'].replace({'Basic':'Undergraduate','2n Cycle':'Undergraduate','Graduation':'Postgraduate','Master':'Postgraduate','PhD':'Postgraduate'})

    #divide the marital status of each customer into two category
    df['Marital_Status']=df['Marital_Status'].replace({'Divorced':'Single','Single':'Single','Married':'In-Relationship','Together':'In-Relationship','Absurd':'Single','Widow':'Single','YOLO':'Single','Alone':'Single'})

    #Remove outlier on customer income
    df= df.drop(df[df.Income>600000].index)

    #divide the income of each customer into 4 categories
    cut_labels_Income = ['Low income', 'Low to medium income', 'Medium to high income', 'High income']
    df['Income_group'] = pd.qcut(df['Income'], q=4, labels=cut_labels_Income)

    #Create Number of Child Column
    df['Number_of_Child']=df['Kidhome']+df['Teenhome']

    #Create total monthly spend column
    df['total_spend']=df['MntWines']+df['MntFruits']+df['MntMeatProducts']+df['MntFishProducts']+df['MntSweetProducts']+df['MntGoldProds']

    #Create total number of accepted campaign column
    df['total_campain_acc']=df['AcceptedCmp1']+df['AcceptedCmp2']+df['AcceptedCmp3']+df['AcceptedCmp4']+df['AcceptedCmp5']+df['Response']

    #Create total purchases
    df['total_purchases']=df['NumWebPurchases']+df['NumStorePurchases']+df['NumCatalogPurchases']

    #Create new data for our model and dropping feature
    model = df.copy()
    model.drop(
        ['Dt_Customer','Kidhome', 'Teenhome', 'Recency', 'ID', 'Year_Birth','Income',''
         'Age','Z_CostContact', 'Z_Revenue'
        ], axis=1, inplace=True)

    #encoding age group feature
    model['Age_group'] = model['Age_group'].replace({'Young': 1, 'Adult': 2 , 'Mature': 3, 'Senior': 4}).astype(int)

    #encoding Education feature
    model['Education'] = model['Education'].replace({'Undergraduate': 1, 'Postgraduate': 2}).astype(int)

    #converting relation feature into int type
    model['Marital_Status'] = model['Marital_Status'].replace({'Single': 1, 'In-Relationship': 2}).astype(int)

    #encoding income group feature
    model['Income_group'] = model['Income_group'].replace({'Low income': 1, 'Low to medium income': 2, 'Medium to high income': 3, 'High income': 4}).astype(int)

    # print(model.head())
    return model