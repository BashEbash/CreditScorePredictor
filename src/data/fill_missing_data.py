import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class NameModeFiller(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x_copy = x.copy()
        x_copy['Name'] = x_copy['Name'].fillna(value=x_copy.groupby("Customer_ID")["Name"].transform(lambda x: x.mode().iloc[0]))
        return x_copy

class SSNTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x_copy = x.copy()

        # Fill missing values in "SSN" based on the mode of each customer's "SSN"
        x_copy["SSN"].fillna(value=x_copy.groupby("Customer_ID")["SSN"].transform(lambda x: x.mode().iloc[0]), inplace=True)

        # Convert "SSN" to integers
        x_copy["SSN"] = x_copy["SSN"].apply(lambda x: int("".join(str(x).split("-"))) if pd.notna(x) else x).astype(np.int64)

        return x_copy

class OccupationTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Fill missing values in "Occupation" based on the mode of each customer's "Occupation"
        X_copy["Occupation"].fillna(value=X_copy.groupby("Customer_ID")["Occupation"].transform(lambda x: x.mode().iloc[0]), inplace=True)

        return X_copy

class MonthlySalaryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_column='Customer_ID', target_column='Monthly_Inhand_Salary'):
        self.mapping_column = mapping_column
        self.target_column = target_column

    def fit(self, X, y=None):
        # Create a dictionary to map Customer_ID to non-null Monthly_Inhand_Salary values
        self.mapping = X.dropna(subset=[self.target_column]).set_index(self.mapping_column)[self.target_column].to_dict()
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Fill missing Monthly_Inhand_Salary values based on Customer_ID
        X_copy[self.target_column] = X_copy.apply(lambda row: self.mapping.get(row[self.mapping_column], row[self.target_column]), axis=1)

        return X_copy

class FillTypeOfLoanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value='NA', target_column='Type_of_Loan'):
        self.fill_value = fill_value
        self.target_column = target_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.target_column].fillna(self.fill_value, inplace=True)
        return X_copy

class InterpolateNumDelayedPaymentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='Num_of_Delayed_Payment'):
        self.target_column = target_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Fill missing values in 'Num_of_Delayed_Payment' using interpolation
        X_copy['Num_of_Delayed_Payment'] = (
            X_copy.groupby('Customer_ID')['Num_of_Delayed_Payment']
            .transform(lambda x: x.interpolate(method='index', limit_direction='both') if x.count() > 1 else x)
        )

        return X_copy

class InterpolateCreditLimitTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='Changed_Credit_Limit'):
        self.target_column = target_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Fill missing values in 'Changed_Credit_Limit' using interpolation
        X_copy['Changed_Credit_Limit'] = (
            X_copy.groupby('Customer_ID')['Changed_Credit_Limit']
            .transform(lambda x: x.interpolate(method='index', limit_direction='both'))
        )

        return X_copy

class InterpolateAndReplaceOutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='Num_Credit_Inquiries', threshold=15):
        self.target_column = target_column
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Identify rows with missing values in 'Num_Credit_Inquiries'
        customer_ids_missing_values = X_copy[X_copy['Num_Credit_Inquiries'].isna()]['Customer_ID'].values

        # Group by 'Customer_ID' and perform index interpolation
        X_copy['Num_Credit_Inquiries'] = (
            X_copy.groupby('Customer_ID')['Num_Credit_Inquiries']
            .transform(lambda x: x.interpolate(method='index', limit_direction='both') if x.count() > 1 else x)
        )

        # Identify rows with 'Num_Credit_Inquiries' > threshold
        filtered_data = X_copy[X_copy['Num_Credit_Inquiries'] > self.threshold]

        # Calculate the mode for each customer
        mode_by_customer = filtered_data.groupby('Customer_ID')['Num_Credit_Inquiries'].transform(lambda x: x.mode().iloc[0])

        # Update the values in the original DataFrame
        X_copy.loc[X_copy['Num_Credit_Inquiries'] > self.threshold, 'Num_Credit_Inquiries'] = mode_by_customer

        return X_copy

class FillCreditMixTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='Credit_Mix'):
        self.target_column = target_column

    def fit(self, X, y=None):
        # Create a dictionary to map Customer_ID to non-null Credit_Mix values
        self.mapping = X.dropna(subset=[self.target_column]).set_index('Customer_ID')[self.target_column].to_dict()
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Fill missing Credit_Mix values based on Customer_ID
        X_copy[self.target_column] = X_copy.apply(lambda row: self.mapping.get(row['Customer_ID'], row[self.target_column]), axis=1)

        return X_copy

class ConvertAndInterpolateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='Credit_History_Age'):
        self.target_column = target_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Convert 'Credit_History_Age' using Month_Converter function
        X_copy[self.target_column] = X_copy[self.target_column].apply(lambda x: self.Month_Converter(x)).astype(np.float64)

        # Identify rows with missing values in 'Credit_History_Age'
        customer_ids_missing_values = X_copy[X_copy[self.target_column].isna()]['Customer_ID'].values

        # Group by 'Customer_ID' and perform index interpolation
        X_copy[self.target_column] = (
            X_copy.groupby('Customer_ID')[self.target_column]
            .transform(lambda x: x.interpolate(method='index', limit_direction='both') if x.count() > 1 else x)
        )

        return X_copy

    def Month_Converter(self, x):
        if pd.notnull(x):
            num1 = int(x.split(' ')[0])
            num2 = int(x.split(' ')[3])
            return (num1 * 12) + num2
        else:
            return x

class InterpolateAmountInvestedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='Amount_invested_monthly'):
        self.target_column = target_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Identify rows with missing values in 'Amount_invested_monthly'
        customer_ids_missing_values = X_copy[X_copy[self.target_column].isna()]['Customer_ID'].values

        # Group by 'Customer_ID' and perform index interpolation
        X_copy[self.target_column] = (
            X_copy.groupby('Customer_ID')[self.target_column]
            .transform(lambda x: x.interpolate(method='index', limit_direction='both') if x.count() > 1 else x)
        )

        return X_copy

class FillPaymentBehaviourTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='Payment_Behaviour'):
        self.target_column = target_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Define a custom function to calculate mode with handling for NaN values
        def custom_mode(series):
            return series.dropna().mode().iloc[0] if not series.dropna().empty else np.nan

        # Calculate and fill missing values with the mode payment behavior for each customer
        X_copy[self.target_column] = (
            X_copy[self.target_column]
            .fillna(X_copy.groupby('Customer_ID')[self.target_column].transform(custom_mode))
        )

        return X_copy

class FillMonthlyBalanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='Monthly_Balance'):
        self.target_column = target_column
        self.imputer = SimpleImputer(strategy='most_frequent')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Group by 'Customer_ID' and perform index interpolation
        X_copy[self.target_column] = (
            X_copy.groupby('Customer_ID')[self.target_column]
            .transform(lambda x: x.interpolate(method='index', limit_direction='both') if x.count() > 1 else x)
        )

        # Impute missing values with the most frequent value
        X_copy[self.target_column] = self.imputer.fit_transform(X_copy[[self.target_column]])

        # Calculate the mode for each 'Monthly_Inhand_Salary'
        customer_mode_payment = (
            X_copy.groupby('Monthly_Inhand_Salary')[self.target_column]
            .transform(lambda x: x.mode().iloc[0])
        )

        # Fill missing values with the corresponding customer's mode payment behavior
        X_copy[self.target_column] = X_copy[self.target_column].fillna(customer_mode_payment)

        return X_copy

def make_dataset(df):
    preprocessor = ColumnTransformer(
        transformers=[
            ('name_mode_filler', NameModeFiller(), ['Name']),
            ('ssn_transformer', SSNTransformer(), ['SSN']),
            ('occupation_transformer', OccupationTransformer(), ['Occupation']),
            ('monthly_salary_transformer', MonthlySalaryTransformer(), ['Monthly_Inhand_Salary']),
            ('fill_type_of_loan_transformer', FillTypeOfLoanTransformer(), ['Type_of_Loan']),
            ('interpolate_num_delayed_payment_transformer', InterpolateNumDelayedPaymentTransformer(),
             ['Num_of_Delayed_Payment']),
            ('interpolate_credit_limit_transformer', InterpolateCreditLimitTransformer(), ['Changed_Credit_Limit']),
            ('interpolate_and_replace_outliers_transformer', InterpolateAndReplaceOutliersTransformer(),
             ['Num_Credit_Inquiries']),
            ('fill_credit_mix_transformer', FillCreditMixTransformer(), ['Credit_Mix']),
            ('convert_and_interpolate_transformer', ConvertAndInterpolateTransformer(), ['Credit_History_Age']),
            ('interpolate_amount_invested_transformer', InterpolateAmountInvestedTransformer(),
             ['Amount_invested_monthly']),
            ('fill_payment_behaviour_transformer', FillPaymentBehaviourTransformer(), ['Payment_Behaviour']),
            ('fill_monthly_balance_transformer', FillMonthlyBalanceTransformer(), ['Monthly_Balance']),
            # Add other transformers for different columns if needed
        ],
        remainder='passthrough'  # Passthrough columns not specified in transformers
    )

    preprocessor.fit_transform(df)

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    make_dataset(df)