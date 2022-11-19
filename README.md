# Schneider-electric-data-science-hackaton Nov 2022

Code for the Nov 2022 Schneider Electric Data Science Hackaton


## Background
The EU contributes 18% of total global warming gas emissions; However, it is increasingly determined to take the lead in the fight against climate change. That is why it has set itself the goal of reaching zero carbon emissions by 2050.

To this end, it has put in place a wealth of resources to help achieve this goal over the next few years, and it will need your help to do so.

## Participants

-   Bruno Moya - 56
-   Joel Poma - 56
-   Marc Alfonso - 56

## Results

By Using agressive data cleaning and parsing, with categorization of variables of the dataset, we achieved a 75%+ F1-Score using Random Forest Model. 

`{'f1_score': 0.7252596753363753, 'precision': 0.7229358290279416, 'recall': 0.7277690113858096, 'accuracy': 0.7025566884796834}`

This are the results of our Cross Validation of the Model (F1 Macro Score):

`Cross-validation scores: [0.74248753 0.74695766 0.74315829 0.74644212 0.80487625]
Average cross-validation score: 0.7567843695113846`

These are our ROC Curves and PR Curves:

![image](https://user-images.githubusercontent.com/10481058/169686805-df3284d7-5e6d-4944-bb1b-215e2abce3a1.png)

![image](https://user-images.githubusercontent.com/10481058/169686900-b85c53d5-0844-42dd-9cca-be3a02517146.png)

## Nuwe Report

![image](https://user-images.githubusercontent.com/10481058/176244071-b081a300-af63-4b77-adf0-9ca6e011bd04.png)


## How to execute the demonstration?

Run the main.py example and install required libraries with PIP:

- `pip install -r requirements.txt`
- `python main.py`
